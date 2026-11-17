//! ★ With a card only: the word memory format's seven tables filled on the
//! card from their rows (`fill_mem_load8` … `fill_memw_w`) are the host fill's
//! words, every cell, padding rows included, through the group upload a block
//! uses (`multilinear::gpu::upload_tables`); the card counts one fill a table;
//! and a row moved later makes other words.
//!
//! Without a card every upload is declined, and these tests say so instead of
//! passing.

use std::sync::Arc;

use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::narrow::NarrowColumns;
use multilinear::records::{Records, RecordsFill, TableRecords, WordRowsFill};

use super::gpack::TraceForm;
use super::word_card::tests::{Rows, rows};
use super::word_card::{
    card_load2w_rows, card_load8_rows, card_loadw_rows, card_memw_w_rows, card_store2w_rows,
    card_store8_rows, card_storew_rows,
};
use super::{mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8, mem_storew, memw_w};

/// One family's table (`family`, [`StreamTable::WORDS`]' order) of `rows`'
/// rows, `height` high, filled on the card or by the host.
///
/// [`StreamTable::WORDS`]: super::trace_builder::windowed::StreamTable::WORDS
struct WordTable {
    family: usize,
    rows: Rows,
    height: usize,
}

const NAMES: [&str; 7] = [
    "MEM_LOAD8",
    "MEM_STORE8",
    "MEM_LOADW",
    "MEM_STOREW",
    "MEM_LOAD2W",
    "MEM_STORE2W",
    "MEMW_W",
];

impl std::fmt::Debug for WordTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} of {}", NAMES[self.family], self.height)
    }
}

impl TableRecords for WordTable {
    fn rows(&self) -> usize {
        self.height
    }

    fn cols(&self) -> usize {
        [
            mem_load8::cols::NUM_COLUMNS,
            mem_store8::cols::NUM_COLUMNS,
            mem_loadw::cols::NUM_COLUMNS,
            mem_storew::cols::NUM_COLUMNS,
            mem_load2w::cols::NUM_COLUMNS,
            mem_store2w::cols::NUM_COLUMNS,
            memw_w::cols::NUM_COLUMNS,
        ][self.family]
    }

    fn fill(&self) -> RecordsFill<'_> {
        let r = &self.rows;
        RecordsFill::Words(match self.family {
            0 => WordRowsFill::Load8(vec![card_load8_rows(&r.load8.0)]),
            1 => WordRowsFill::Store8(vec![card_store8_rows(&r.store8.0)]),
            2 => WordRowsFill::LoadW(vec![card_loadw_rows(&r.loadw.0)]),
            3 => WordRowsFill::StoreW(vec![card_storew_rows(&r.storew.0)]),
            4 => WordRowsFill::Load2W(vec![card_load2w_rows(&r.load2w.0)]),
            5 => WordRowsFill::Store2W(vec![card_store2w_rows(&r.store2w.0)]),
            _ => WordRowsFill::MemwW(vec![card_memw_w_rows(&r.memw_w.0)]),
        })
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        let r = &self.rows;
        let f = TraceForm::Narrow;
        let mut trace = match self.family {
            0 => mem_load8::generate_mem_load8_trace_segments_as(&[&r.load8.0], f),
            1 => mem_store8::generate_mem_store8_trace_segments_as(&[&r.store8.0], f),
            2 => mem_loadw::generate_mem_loadw_trace_segments_as(&[&r.loadw.0], f),
            3 => mem_storew::generate_mem_storew_trace_segments_as(&[&r.storew.0], f),
            4 => mem_load2w::generate_mem_load2w_trace_segments_as(&[&r.load2w.0], f),
            5 => mem_store2w::generate_mem_store2w_trace_segments_as(&[&r.store2w.0], f),
            _ => memw_w::generate_memw_w_trace_segments_as(&[&r.memw_w.0], f),
        };
        trace.pack_main_narrow();
        trace.take_narrow_main()
    }
}

fn host_words(records: &Records) -> Vec<Vec<u64>> {
    let packed = records.host_fill().expect("the host fills it");
    (0..packed.cols()).map(|c| packed.column(c)).collect()
}

fn mismatches(got: &[Vec<u64>], want: &[Vec<u64>]) -> usize {
    assert_eq!(got.len(), want.len(), "the columns");
    got.iter()
        .zip(want)
        .map(|(g, w)| g.iter().zip(w).filter(|(a, b)| a != b).count())
        .sum()
}

fn card_words(what: &str, records: &Records) -> Vec<Vec<u64>> {
    upload_tables::<F>(&[TableColumns::Records(records)])
        .and_then(|s| s.download())
        .unwrap_or_else(|| panic!("{what}: the card takes the table"))
}

/// ★ Each family's whole and padded tables filled on the card are the host
/// fill's words; the card counts one fill a table.
#[test]
fn the_card_fills_the_word_tables_as_the_host_does() {
    let mut cases = 0usize;
    for (k, (n, height)) in [
        (4usize, 4usize),
        (5, 8),
        (1000, 1024),
        (4096, 4096),
        (70_000, 1 << 17),
    ]
    .into_iter()
    .enumerate()
    {
        for (family, name) in NAMES.iter().enumerate() {
            let records = Records(Arc::new(WordTable {
                family,
                rows: rows(500 + 10 * k as u64 + family as u64, n),
                height,
            }));
            let what = format!("{name} {n} of {height}");
            let (before, _) = multilinear::records::fills();
            let card = card_words(&what, &records);
            let (after, _) = multilinear::records::fills();
            assert_eq!(after, before + 1, "{what}: filled on the card");
            assert_eq!(mismatches(&card, &host_words(&records)), 0, "{what}");
            println!("CARD WORD FILL {what}: 0 words differ");
            cases += 1;
        }
    }
    println!("CARD WORD FILL VERDICT PASS cases={cases}");
}

/// A row moved later on the card makes other words than the original's host
/// fill, in every family.
#[test]
fn a_moved_word_row_fills_other_words() {
    for (family, &what) in NAMES.iter().enumerate() {
        let original = rows(77, 64);
        let want = host_words(&Records(Arc::new(WordTable {
            family,
            rows: rows(77, 64),
            height: 64,
        })));
        let mut moved = original;
        match family {
            0 => moved.load8.0[3] = moved.load8.0[3].later(4),
            1 => moved.store8.0[3] = moved.store8.0[3].later(4),
            2 => moved.loadw.0[3] = moved.loadw.0[3].later(4),
            3 => moved.storew.0[3] = moved.storew.0[3].later(4),
            4 => moved.load2w.0[3] = moved.load2w.0[3].later(4),
            5 => moved.store2w.0[3] = moved.store2w.0[3].later(4),
            _ => moved.memw_w.0[3] = moved.memw_w.0[3].later(4),
        }
        let records = Records(Arc::new(WordTable {
            family,
            rows: moved,
            height: 64,
        }));
        let n = mismatches(&card_words(what, &records), &want);
        println!("CARD WORD FILL MUTANT {what}: {n} words differ");
        assert!(n > 0, "{what}: the moved row moved no word");
    }
}
