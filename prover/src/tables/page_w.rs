//! PAGE_W: a page's words' first and last tokens, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! One table a page of [`DEFAULT_PAGE_SIZE`] bytes, as PAGE's page grid, one
//! row an aligned 8-byte word: [`ROWS`] rows. A row produces its word's first
//! token, `(page_base_lo + 8·OFFSET, page_base_hi, 0, init)`, and consumes its
//! last, `(same address, TIMESTAMP, fini)`; a word nothing touches has
//! `TIMESTAMP = 0` and `fini = init`, and its two tokens cancel.
//!
//! PAGE_W is the only table that mints word tokens
//! ([`super::word_memory`]). `OFFSET` is preprocessed on every page kind, the
//! row index `0..ROWS`, and the base is page-aligned, so every address it names
//! is an integer multiple of 8 below `2^32`: a token exists at no other
//! address, which is what refuses an unaligned access anywhere else.
//!
//! The initial word is four 16-bit halves, `INIT_0..INIT_3`, low first. On an
//! ELF or a zero page they are preprocessed (the verifier's own words, so
//! canonical). On a private-input page they hold the input and are main
//! columns, each sent to `IS_HALF`, so the word they make is canonical too.
//! The final word is two 32-bit halves and is not range-checked: it is the
//! word the last access produced, and every producer keeps its words canonical
//! (`super::word_memory`).
//!
//! ## Columns (8)
//!
//! `OFFSET`, `INIT_0..INIT_3`, `FINI_LO`, `FINI_HI`, `TIMESTAMP`; the
//! preprocessed prefix is the first [`NUM_PREPROCESSED_COLS`] on an ELF or a
//! zero page, `OFFSET` alone ([`NUM_PREPROCESSED_COLS_PRIVATE`]) on a
//! private-input page.
//!
//! ## Interactions (2, or 6 on a private-input page)
//!
//! - the first token produced and the last consumed, at 1 on every row;
//! - on a private-input page, `IS_HALF` on each `INIT` half, at 1.

use std::collections::HashMap;

use stark::config::Commitment;
use stark::leaf_layout::LeafLayout;
use stark::lookup::{BusInteraction, BusValue, LazyCommitment, LinearTerm, Multiplicity};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_memory::{WORD_BYTES, column, halves, term, token};

/// Rows of a PAGE_W table: one a word of a page.
pub const ROWS: usize = DEFAULT_PAGE_SIZE / WORD_BYTES as usize;

/// A PAGE_W table's height in variables: every PAGE_W table is exactly
/// [`ROWS`] rows, and the block verifier holds a stated height to it.
pub const NUM_VARS: usize = ROWS.trailing_zeros() as usize;

/// The name prefix of every PAGE_W table (`PAGE_W:0x<base>`).
pub const NAME_PREFIX: &str = "PAGE_W:";

/// Column definitions for the PAGE_W table.
pub mod cols {
    /// The word's index in the page, `0..ROWS` (preprocessed).
    pub const OFFSET: usize = 0;
    /// The initial word's four 16-bit halves, low first.
    pub const INIT: [usize; 4] = [1, 2, 3, 4];
    /// The final word's low and high 32-bit halves.
    pub const FINI_LO: usize = 5;
    pub const FINI_HI: usize = 6;
    /// The final token's timestamp (0 if the word is never touched).
    pub const TIMESTAMP: usize = 7;
    pub const NUM_COLUMNS: usize = 8;
}

/// Preprocessed columns of an ELF or a zero page: `OFFSET` and the four
/// `INIT` halves.
pub const NUM_PREPROCESSED_COLS: usize = 5;

/// Preprocessed columns of a private-input page: `OFFSET` alone. The input's
/// halves stay main columns the verifier never recomputes; `OFFSET` is the
/// row's address and must never be the prover's.
pub const NUM_PREPROCESSED_COLS_PRIVATE: usize = 1;

/// Final state of one word.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FinalWordState {
    /// The last access's timestamp (0 if never accessed).
    pub timestamp: u64,
    /// The word after it.
    pub value: u64,
}

/// Map from an aligned word address to its final state.
pub type FinalWordMap = HashMap<u64, FinalWordState>;

/// Whether `page_base` can be a PAGE_W page: page-aligned. Then every address
/// the page names, `page_base_lo + 8·OFFSET`, is below `2^32` (the base's low
/// word is at most `2^32 − DEFAULT_PAGE_SIZE`) and never carries into the high
/// word.
pub fn check_page_base(page_base: u64) -> Result<(), String> {
    if page_base.is_multiple_of(DEFAULT_PAGE_SIZE as u64) {
        Ok(())
    } else {
        Err(format!(
            "a PAGE_W page is page-aligned, not at {page_base:#x}"
        ))
    }
}

// =========================================================================
// Page runs: the page tables a block holds
// =========================================================================

/// The most pages a page block holds, as a power of two: a verifier constant
/// bound, so a block is at most `2^(15 + 6)` rows and its span `2^24` bytes,
/// far below the `2^32` a block's span must divide.
pub const MAX_PAGE_RUN_LOG2: u8 = 6;

/// The name prefix of a zero block (`PAGE_Z:0x<base>:<pages>`).
pub const ZERO_NAME_PREFIX: &str = "PAGE_Z:";

/// The name prefix of a private-input block (`PAGE_P:0x<base>:<pages>`).
pub const PRIVATE_NAME_PREFIX: &str = "PAGE_P:";

/// What a page table holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PageTableKind {
    /// One page, a PAGE_W table of its own ([`NAME_PREFIX`]): every page when
    /// page runs are off, and a page of initialized data when they are on.
    Page,
    /// A block of zero-initialized pages ([`ZERO_NAME_PREFIX`]): no `INIT`
    /// column, every word it mints the constant 0.
    Zero,
    /// A block of private-input pages ([`PRIVATE_NAME_PREFIX`]): a
    /// private-input page's columns at the block's height.
    Private,
}

/// One page table: `2^log_pages` consecutive pages from `base`, the page
/// configs `first..first + 2^log_pages` of the list it was cut from
/// ([`page_tables`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageTable {
    pub kind: PageTableKind,
    pub base: u64,
    pub log_pages: u8,
    pub first: usize,
}

impl PageTable {
    /// Its pages.
    pub fn pages(&self) -> usize {
        1 << self.log_pages
    }

    /// Its height in variables: [`NUM_VARS`] a page.
    pub fn num_vars(&self) -> usize {
        NUM_VARS + usize::from(self.log_pages)
    }

    /// Its rows: one a word of its pages.
    pub fn rows(&self) -> usize {
        1 << self.num_vars()
    }

    /// The bytes it covers.
    pub fn span(&self) -> u64 {
        (DEFAULT_PAGE_SIZE as u64) << self.log_pages
    }

    /// Its table's name: a page's `PAGE_W:0x<base>`, a block's
    /// `PAGE_Z:0x<base>:<pages>` or `PAGE_P:0x<base>:<pages>`.
    pub fn name(&self) -> String {
        match self.kind {
            PageTableKind::Page => format!("{NAME_PREFIX}0x{:x}", self.base),
            PageTableKind::Zero => {
                format!("{ZERO_NAME_PREFIX}0x{:x}:{}", self.base, self.pages())
            }
            PageTableKind::Private => {
                format!("{PRIVATE_NAME_PREFIX}0x{:x}:{}", self.base, self.pages())
            }
        }
    }

    /// Its pages' configs in `configs` (the list it was cut from), or the
    /// reason they are not its pages: past the list, or a page whose base or
    /// class ([`page_kind`]) is not the table's.
    pub fn configs<'a>(&self, configs: &'a [PageConfig]) -> Result<&'a [PageConfig], String> {
        let mine = self
            .first
            .checked_add(self.pages())
            .and_then(|end| configs.get(self.first..end))
            .ok_or_else(|| format!("{} is past the page list", self.name()))?;
        for (k, config) in mine.iter().enumerate() {
            let base = (k as u64)
                .checked_mul(DEFAULT_PAGE_SIZE as u64)
                .and_then(|offset| self.base.checked_add(offset))
                .ok_or_else(|| format!("{} runs past the address space", self.name()))?;
            let kind = match self.kind {
                // A page table of one page takes the page whatever its class
                // (page runs off: every page is one).
                PageTableKind::Page => PageTableKind::Page,
                _ => page_kind(config),
            };
            if config.page_base != base || kind != self.kind {
                return Err(format!(
                    "{} holds page {k} at {base:#x}, not the {:?} page at {:#x}",
                    self.name(),
                    page_kind(config),
                    config.page_base
                ));
            }
        }
        Ok(mine)
    }
}

/// The class a page takes under page runs: a private-input page's is
/// [`PageTableKind::Private`]; a page with a nonzero initial byte (the
/// program's data) is a table of its own, [`PageTableKind::Page`]; every
/// other page (a runtime page, an all-zero page of the program) is
/// [`PageTableKind::Zero`]. It reads the private flag first and no private
/// byte, so the prover's configs and the verifier's give the same class.
pub fn page_kind(config: &PageConfig) -> PageTableKind {
    if config.is_private_input {
        PageTableKind::Private
    } else if config
        .init_values
        .as_ref()
        .is_some_and(|values| values.iter().any(|&b| b != 0))
    {
        PageTableKind::Page
    } else {
        PageTableKind::Zero
    }
}

/// Whether `table` can be a page table: at most [`MAX_PAGE_RUN_LOG2`] pages
/// as a power of two, and its base a multiple of its span. Then every address
/// it names, `base_lo + 8·OFFSET`, is below `2^32` (the span divides `2^32`,
/// so the block never crosses a multiple of it) and never carries into the
/// high word.
pub fn check_page_table(table: &PageTable) -> Result<(), String> {
    if table.log_pages > MAX_PAGE_RUN_LOG2 {
        return Err(format!(
            "{} holds 2^{} pages, a page table at most 2^{MAX_PAGE_RUN_LOG2}",
            table.name(),
            table.log_pages
        ));
    }
    if !table.base.is_multiple_of(table.span()) {
        return Err(format!(
            "{} is not aligned to its span {:#x}",
            table.name(),
            table.span()
        ));
    }
    Ok(())
}

/// ★ The page-run rule: the page tables of `configs` (a block's page list,
/// which the verifier derives from the program and the statement: sorted, one
/// page a base, every base page-aligned) when a run of pages takes at most
/// `2^run_log2` pages a table. A verifier constant computes it from public
/// inputs alone, never from a proof.
///
/// - `run_log2 = 0`: one [`PageTableKind::Page`] table a page, today's
///   tables.
/// - Otherwise each page is classed ([`page_kind`]); a page of data is a
///   table of its own, and each maximal run of contiguous pages of one other
///   class (zero, or private input) is cut from its first page into the
///   largest **aligned** blocks that fit: a block of `2^c` pages, `c ≤
///   run_log2`, starts at a page index divisible by `2^c`.
///
/// The tables are a partition of the list in its order. Refuses a
/// `run_log2` past [`MAX_PAGE_RUN_LOG2`], and a list that is not sorted
/// strictly by base or holds an unaligned base.
pub fn page_tables(configs: &[PageConfig], run_log2: u8) -> Result<Vec<PageTable>, String> {
    if run_log2 > MAX_PAGE_RUN_LOG2 {
        return Err(format!(
            "page runs of 2^{run_log2} pages: at most 2^{MAX_PAGE_RUN_LOG2}"
        ));
    }
    for config in configs {
        check_page_base(config.page_base)?;
    }
    if let Some(w) = configs
        .windows(2)
        .find(|w| w[0].page_base >= w[1].page_base)
    {
        return Err(format!(
            "the page list is not sorted strictly by base: {:#x} then {:#x}",
            w[0].page_base, w[1].page_base
        ));
    }
    let page = DEFAULT_PAGE_SIZE as u64;
    let single = |first: usize| PageTable {
        kind: PageTableKind::Page,
        base: configs[first].page_base,
        log_pages: 0,
        first,
    };
    if run_log2 == 0 {
        return Ok((0..configs.len()).map(single).collect());
    }
    let mut tables = Vec::new();
    let mut i = 0;
    while i < configs.len() {
        let kind = page_kind(&configs[i]);
        if kind == PageTableKind::Page {
            tables.push(single(i));
            i += 1;
            continue;
        }
        // The run: the pages of this class each one page past the last.
        let mut end = i + 1;
        while end < configs.len()
            && page_kind(&configs[end]) == kind
            && configs[end - 1].page_base.checked_add(page) == Some(configs[end].page_base)
        {
            end += 1;
        }
        // Cut from its first page, each block the largest aligned one left.
        let mut at = i;
        while at < end {
            let index = configs[at].page_base / page;
            let left = end - at;
            let mut log = 0u8;
            while log < run_log2 && (2usize << log) <= left && index.is_multiple_of(2u64 << log) {
                log += 1;
            }
            tables.push(PageTable {
                kind,
                base: configs[at].page_base,
                log_pages: log,
                first: at,
            });
            at += 1 << log;
        }
        i = end;
    }
    Ok(tables)
}

/// The initial word at `word` (its index in the page): the config's eight
/// bytes there, little-endian, a missing byte zero.
pub fn init_word(config: &PageConfig, word: usize) -> u64 {
    let mut bytes = [0u8; 8];
    if let Some(values) = config.init_values.as_ref() {
        let start = word * WORD_BYTES as usize;
        for (k, byte) in bytes.iter_mut().enumerate() {
            *byte = values.get(start + k).copied().unwrap_or(0);
        }
    }
    u64::from_le_bytes(bytes)
}

/// A word's four 16-bit halves, low first.
#[inline]
fn quarters(word: u64) -> [u16; 4] {
    [
        word as u16,
        (word >> 16) as u16,
        (word >> 32) as u16,
        (word >> 48) as u16,
    ]
}

// =========================================================================
// Trace generation
// =========================================================================

/// The PAGE_W trace of the page `config`, its words' final states read from
/// `finals` (a word missing from it is untouched: timestamp 0, its initial
/// word). Refuses a base [`check_page_base`] refuses.
pub fn generate_page_w_trace(
    config: &PageConfig,
    finals: &FinalWordMap,
) -> Result<TraceTable<GoldilocksField, GoldilocksExtension>, String> {
    page_w_trace_of(config, |row, init| {
        let address = config.page_base + row as u64 * WORD_BYTES;
        finals.get(&address).copied().unwrap_or(FinalWordState {
            timestamp: 0,
            value: init,
        })
    })
}

/// The PAGE_W trace of the page `config` from its bytes' final cells
/// (`final_page`: each byte's value and last-access timestamp, the page's
/// [`DEFAULT_PAGE_SIZE`] of them; `None` for a page no byte of which was
/// set): a word's final value is its eight bytes and its timestamp the
/// largest of theirs, a word whose bytes are all at 0 untouched (its initial
/// word). Refuses a base [`check_page_base`] refuses, or a page that is not
/// the page's size.
pub fn generate_page_w_trace_from_bytes(
    config: &PageConfig,
    final_page: Option<&[(u8, u64)]>,
) -> Result<TraceTable<GoldilocksField, GoldilocksExtension>, String> {
    if let Some(page) = final_page
        && page.len() != DEFAULT_PAGE_SIZE
    {
        return Err(format!(
            "a page's final cells are {DEFAULT_PAGE_SIZE} bytes, not {}",
            page.len()
        ));
    }
    page_w_trace_of(config, |row, init| {
        let Some(page) = final_page else {
            return FinalWordState {
                timestamp: 0,
                value: init,
            };
        };
        let cells = &page[row * WORD_BYTES as usize..(row + 1) * WORD_BYTES as usize];
        let timestamp = cells.iter().map(|&(_, ts)| ts).max().unwrap_or(0);
        if timestamp == 0 {
            return FinalWordState {
                timestamp: 0,
                value: init,
            };
        }
        let value = cells
            .iter()
            .enumerate()
            .fold(0u64, |word, (k, &(byte, _))| {
                word | u64::from(byte) << (8 * k)
            });
        FinalWordState { timestamp, value }
    })
}

/// The PAGE_W trace of the page `config`, each row's final state
/// `final_of(row, initial word)`.
fn page_w_trace_of(
    config: &PageConfig,
    final_of: impl Fn(usize, u64) -> FinalWordState,
) -> Result<TraceTable<GoldilocksField, GoldilocksExtension>, String> {
    check_page_base(config.page_base)?;
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(ROWS * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;
    for row in 0..ROWS {
        table.set_u64(row, cols::OFFSET, row as u64);
        let init = init_word(config, row);
        table.set_halves(row, cols::INIT[0], &quarters(init));
        let state = final_of(row, init);
        let (fini_lo, fini_hi) = halves(state.value);
        table.set_word(row, cols::FINI_LO, fini_lo);
        table.set_word(row, cols::FINI_HI, fini_hi);
        table.set_u64(row, cols::TIMESTAMP, state.timestamp);
    }
    Ok(trace)
}

/// The BITWISE lookups a page sends: on a private-input page, `IS_HALF` on
/// each of every row's four `INIT` halves; none on any other page.
pub fn bitwise_ops(config: &PageConfig) -> impl Iterator<Item = BitwiseOperation> + '_ {
    let rows = if config.is_private_input { ROWS } else { 0 };
    (0..rows).flat_map(move |row| {
        quarters(init_word(config, row)).into_iter().map(|half| {
            BitwiseOperation::halfword(BitwiseOperationType::IsHalf, half as u8, (half >> 8) as u8)
        })
    })
}

// =========================================================================
// Preprocessed columns
// =========================================================================

/// `OFFSET` alone: `0..ROWS`, the same for every page.
pub fn offset_column() -> Vec<FE> {
    (0..ROWS as u64).map(FE::from).collect()
}

/// The preprocessed columns of an ELF or a zero page: `OFFSET` and the four
/// halves of each initial word, all known to the verifier from the ELF.
pub fn preprocessed_columns(config: &PageConfig) -> Vec<Vec<FE>> {
    let mut init: [Vec<FE>; 4] = std::array::from_fn(|_| Vec::with_capacity(ROWS));
    for row in 0..ROWS {
        for (column, half) in init.iter_mut().zip(quarters(init_word(config, row))) {
            column.push(FE::from(u64::from(half)));
        }
    }
    let mut columns = vec![offset_column()];
    columns.extend(init);
    columns
}

/// A page's preprocessed columns, by kind: `OFFSET` alone on a private-input
/// page, `OFFSET` and the initial halves on any other.
pub fn preprocessed_columns_of(config: &PageConfig) -> Vec<Vec<FE>> {
    if config.is_private_input {
        vec![offset_column()]
    } else {
        preprocessed_columns(config)
    }
}

/// The number of preprocessed columns of `config`'s page.
pub fn num_preprocessed_cols(config: &PageConfig) -> usize {
    if config.is_private_input {
        NUM_PREPROCESSED_COLS_PRIVATE
    } else {
        NUM_PREPROCESSED_COLS
    }
}

/// The page's preprocessed root under `layout`, from [`preprocessed_columns_of`].
pub fn commitment(config: &PageConfig, options: &ProofOptions, layout: LeafLayout) -> Commitment {
    super::page::commit_preprocessed_columns(
        &preprocessed_columns_of(config),
        ROWS,
        options,
        layout,
    )
}

/// The page's preprocessed commitment for both leaf layouts, each computed on
/// first use ([`commitment`]).
pub fn lazy_commitment(config: &PageConfig, options: &ProofOptions) -> LazyCommitment {
    let (c, o) = (config.clone(), options.clone());
    let base = LazyCommitment::deferred(move || commitment(&c, &o, LeafLayout::RowPair));
    let (c, o) = (config.clone(), options.clone());
    base.with_one_row(move || Some(commitment(&c, &o, LeafLayout::Row)))
}

// =========================================================================
// Bus interactions
// =========================================================================

/// The interactions of the PAGE_W table of the page at `page_base`
/// (`private`: a private-input page), or the reason there is none
/// ([`check_page_base`]): the first token produced, the last consumed, and on
/// a private-input page the four `IS_HALF` on `INIT`.
pub fn bus_interactions(page_base: u64, private: bool) -> Result<Vec<BusInteraction>, String> {
    check_page_base(page_base)?;
    let (base_lo, base_hi) = halves(page_base);
    let address = || {
        [
            BusValue::linear(vec![
                LinearTerm::Constant(i64::from(base_lo)),
                term(WORD_BYTES as i64, cols::OFFSET),
            ]),
            BusValue::constant(u64::from(base_hi)),
        ]
    };
    let [h0, h1, h2, h3] = cols::INIT;
    let mut interactions = vec![
        BusInteraction::receiver(
            BusId::MemWord,
            Multiplicity::One,
            token(
                address(),
                BusValue::constant(0),
                [
                    BusValue::linear(vec![term(1, h0), term(1 << 16, h1)]),
                    BusValue::linear(vec![term(1, h2), term(1 << 16, h3)]),
                ],
            ),
        ),
        BusInteraction::sender(
            BusId::MemWord,
            Multiplicity::One,
            token(
                address(),
                column(cols::TIMESTAMP),
                [column(cols::FINI_LO), column(cols::FINI_HI)],
            ),
        ),
    ];
    if private {
        interactions.extend(cols::INIT.map(|half| {
            BusInteraction::sender(BusId::IsHalfword, Multiplicity::One, vec![column(half)])
        }));
    }
    Ok(interactions)
}
