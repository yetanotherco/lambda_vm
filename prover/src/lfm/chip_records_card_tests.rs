//! ★ The card fills an LFM chip table from its records
//! ([`super::chip_records::LfmChipRecords`]) with the words the host fill
//! writes, word for word: every chip of a program that has them all, cut into
//! chunk tables (a chunk's offset into the chip's records, its padding rows),
//! `LFM_BITDEC`'s `Z = 1` row and its `GINV`s; alone, as a group's upload, and
//! through the one-table path (`math_cuda::fill::lfm_chip_to_host`).
//!
//! Card only: without a device every upload is declined, and these tests say
//! so instead of passing.

use std::sync::Arc;

use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::records::{LfmValues, Records, RecordsFill, TableRecords};

use super::chip_records::{ChipKind, LfmChipRecords, LfmChipValues};
use super::chip_records_tests::every_chip;
use super::executor::execute;
use super::wlfm::{WlfmRate, build_wlfm, test_format};
use super::wlfm_tests::{HASHER, arenas};

/// Every chip table of [`every_chip`] cut at 2^4 rows, held as records: at
/// most the first two and the last of a chip's tables (`LFM_RANGE` is 4,096
/// of them), named.
fn cases() -> Vec<(String, LfmChipRecords)> {
    let _cut = test_format::set_chunk_log(4);
    let program = every_chip(37, 77);
    let mut built = build_wlfm(&program, HASHER, WlfmRate::Two).expect("builds");
    built.hold_chip_prefixes().expect("the prefixes pack");
    let mut exec = execute(&program, &arenas(), &HASHER).expect("executes");
    let values = LfmChipValues::take(&mut exec.records);
    let a = &built.artifacts;
    let airs = a.airs();
    let mut t = 0usize;
    let mut out = Vec::new();
    for (i, air) in airs.lfm.air_refs().into_iter().enumerate() {
        let chunks = &a.lfm_chunks[i];
        let Some(kind) = ChipKind::of_air(air.name()) else {
            t += chunks.len();
            continue;
        };
        let mut start = 0usize;
        for (k, &rows) in chunks.iter().enumerate() {
            if k < 2 || k + 1 == chunks.len() {
                let prefix = built.chip_prefixes()[t].clone().expect("a held prefix");
                let records = values
                    .table(kind, prefix, start, rows)
                    .expect("held")
                    .expect("the records fit");
                out.push((format!("{kind:?} table {k} of {}", chunks.len()), records));
            }
            start += rows;
            t += 1;
        }
    }
    out
}

/// The card's math-cuda kind for the records'.
fn card_kind(kind: LfmValues) -> math_cuda::fill::LfmValues {
    match kind {
        LfmValues::Words(n) => math_cuda::fill::LfmValues::Words(n),
        LfmValues::Bits => math_cuda::fill::LfmValues::Bits,
        LfmValues::Zeros(n) => math_cuda::fill::LfmValues::Zeros(n),
    }
}

#[test]
#[ignore = "card: fills the chip tables on the GPU; run on the box with --ignored"]
fn the_card_fills_the_chip_records_word_for_word() {
    let cases = cases();
    assert!(cases.len() >= ChipKind::ALL.len());
    let (card_before, _) = multilinear::records::fills();
    let mut group: Vec<Records> = Vec::new();
    let mut group_host: Vec<Vec<u64>> = Vec::new();
    for (what, records) in cases {
        let host = records.host_columns();
        // The one-table path.
        let RecordsFill::LfmChip {
            prefix,
            values,
            kind,
            real,
        } = records.fill()
        else {
            panic!("{what}: an LFM chip fill");
        };
        let input = math_cuda::narrow::NarrowInput::new(
            prefix.data(),
            prefix.offsets(),
            prefix.widths(),
            prefix.rows(),
            prefix.cols(),
        );
        let words = math_cuda::fill::lfm_chip_to_host(input, values, card_kind(kind), real)
            .unwrap_or_else(|e| panic!("{what}: the card fills it: {e:?}"));
        let rows = records.rows();
        for (c, column) in host.iter().enumerate() {
            assert_eq!(
                &words[c * rows..(c + 1) * rows],
                column.as_slice(),
                "{what}: one-table fill, column {c}"
            );
        }
        // A group's upload of this table alone.
        let records = Records(Arc::new(records));
        let card = upload_tables::<F>(&[TableColumns::Records(&records)])
            .and_then(|store| store.download())
            .unwrap_or_else(|| panic!("{what}: the card takes it"));
        assert_eq!(card, host, "{what}: a group's upload");
        group_host.extend(host);
        group.push(records);
    }
    // Every table in one group's upload: each at its place in the store.
    let tables: Vec<TableColumns<'_, F>> = group.iter().map(TableColumns::Records).collect();
    let card = upload_tables::<F>(&tables)
        .and_then(|store| store.download())
        .expect("the card takes the group");
    assert_eq!(card, group_host, "the group's store");
    let (card_after, _) = multilinear::records::fills();
    assert!(
        card_after - card_before >= 2 * group.len() as u64,
        "every upload filled on the card: {} fills",
        card_after - card_before
    );
}

/// A fill whose values land a row late is caught word by word, for a chip of
/// value words and for the bit decomposition: the comparison above is not
/// vacuous.
#[cfg(feature = "test-cuda-faults")]
#[test]
#[ignore = "card: fills the chip tables on the GPU; run on the box with --ignored"]
fn a_chip_fill_a_row_late_is_caught() {
    use std::sync::atomic::Ordering::Relaxed;
    for (what, records) in cases()
        .into_iter()
        .filter(|(what, _)| what.starts_with("Balu") || what.starts_with("Bitdec"))
    {
        let host = records.host_columns();
        let records = Records(Arc::new(records));
        math_cuda::fill::FAULT_RECORDS_SKIP.store(1, Relaxed);
        let card = upload_tables::<F>(&[TableColumns::Records(&records)])
            .and_then(|store| store.download());
        math_cuda::fill::FAULT_RECORDS_SKIP.store(0, Relaxed);
        let card = card.unwrap_or_else(|| panic!("{what}: the card takes it"));
        assert_ne!(card, host, "{what}: a row late, and every word the host's");
    }
}
