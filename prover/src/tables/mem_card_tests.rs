//! ★ Box only (a card): the fused memory tables (MEM_LOAD, MEM_STORE) filled on
//! the card from their records (`fill_mem_load`, `fill_mem_store`) are the
//! host fill's words, every cell, padding rows included, through the group
//! upload a block uses (`multilinear::gpu::upload_tables`); the card counts one
//! fill a table and the host none; and a mutant record makes other words.
//!
//! Without a card every upload is declined, and these tests say so instead of
//! passing.

use std::sync::Arc;

use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::narrow::NarrowColumns;
use multilinear::records::{Records, RecordsFill, TableRecords};

use super::gpack::TraceForm;
use super::mem_card::{card_load_rows, card_store_rows};
use super::mem_load::{MemLoadRow, generate_mem_load_trace_segments_as};
use super::mem_store::{MemStoreRow, generate_mem_store_trace_segments_as};

/// A table held as its fused rows, filled on the card or by the host.
enum Fused {
    Load(Vec<MemLoadRow>, usize),
    Store(Vec<MemStoreRow>, usize),
}

impl std::fmt::Debug for Fused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(rows, h) => write!(f, "MEM_LOAD rows: {} of {h}", rows.len()),
            Self::Store(rows, h) => write!(f, "MEM_STORE rows: {} of {h}", rows.len()),
        }
    }
}

impl TableRecords for Fused {
    fn rows(&self) -> usize {
        match self {
            Self::Load(_, h) | Self::Store(_, h) => *h,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Load(..) => super::mem_load::cols::NUM_COLUMNS,
            Self::Store(..) => super::mem_store::cols::NUM_COLUMNS,
        }
    }

    fn fill(&self) -> RecordsFill<'_> {
        match self {
            Self::Load(rows, _) => RecordsFill::MemLoad {
                rows: vec![card_load_rows(rows)],
            },
            Self::Store(rows, _) => RecordsFill::MemStore {
                rows: vec![card_store_rows(rows)],
            },
        }
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        let mut trace = match self {
            Self::Load(rows, _) => generate_mem_load_trace_segments_as(&[rows], TraceForm::Narrow),
            Self::Store(rows, _) => {
                generate_mem_store_trace_segments_as(&[rows], TraceForm::Narrow)
            }
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

/// The cases: whole and padded tables of every width, sign and step.
fn cases() -> Vec<(String, Records)> {
    let mut out = Vec::new();
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
        let loads = super::mem_card::tests::load_rows(300 + k as u64, n);
        let stores = super::mem_card::tests::store_rows(400 + k as u64, n);
        out.push((
            format!("MEM_LOAD {n} of {height}"),
            Records(Arc::new(Fused::Load(loads, height))),
        ));
        out.push((
            format!("MEM_STORE {n} of {height}"),
            Records(Arc::new(Fused::Store(stores, height))),
        ));
    }
    out
}

/// ★ Each case filled on the card is the host fill's words; the card counts
/// one fill a table, the host none for it.
#[test]
fn the_card_fills_the_fused_memory_tables_as_the_host_does() {
    let cases = cases();
    for (what, records) in &cases {
        let (card_before, _) = multilinear::records::fills();
        let card = upload_tables::<F>(&[TableColumns::Records(records)])
            .and_then(|s| s.download())
            .unwrap_or_else(|| panic!("{what}: the card takes the table"));
        let (card_after, _) = multilinear::records::fills();
        assert_eq!(card_after, card_before + 1, "{what}: filled on the card");
        assert_eq!(mismatches(&card, &host_words(records)), 0, "{what}");
        println!("CARD MEM FILL {what}: 0 words differ");
    }
    println!("CARD MEM FILL VERDICT PASS cases={}", cases.len());
}

/// A record changed (a load's width, a store's old bytes) makes other words
/// on the card than the original's host fill.
#[test]
fn a_changed_fused_record_fills_other_words() {
    let mut loads = super::mem_card::tests::load_rows(7, 64);
    let mut stores = super::mem_card::tests::store_rows(8, 64);
    let want_l = host_words(&Records(Arc::new(Fused::Load(loads.clone(), 64))));
    let want_s = host_words(&Records(Arc::new(Fused::Store(stores.clone(), 64))));
    let i = loads
        .iter()
        .position(|r| r.width() == 8)
        .expect("an 8-byte load");
    let r = loads[i];
    loads[i] = MemLoadRow::new(
        r.address(),
        r.timestamp(),
        r.old_timestamp(),
        4,
        r.signed(),
        r.res(),
    )
    .expect("a 4-byte load");
    let s = stores[0];
    stores[0] = MemStoreRow::new(
        s.address(),
        s.timestamp(),
        s.old_timestamp(),
        s.width(),
        s.value(),
        !s.old(),
    )
    .expect("the store");
    for (what, records, want) in [
        (
            "MEM_LOAD",
            Records(Arc::new(Fused::Load(loads, 64))),
            want_l,
        ),
        (
            "MEM_STORE",
            Records(Arc::new(Fused::Store(stores, 64))),
            want_s,
        ),
    ] {
        let card = upload_tables::<F>(&[TableColumns::Records(&records)])
            .and_then(|s| s.download())
            .unwrap_or_else(|| panic!("{what}: the card takes the table"));
        let n = mismatches(&card, &want);
        println!("CARD MEM FILL MUTANT {what}: {n} words differ");
        assert!(n > 0, "{what}: the changed record moved no word");
    }
}
