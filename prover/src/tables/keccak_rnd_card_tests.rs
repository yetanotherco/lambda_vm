//! ★ The card fills a KECCAK_RND table from its records
//! ([`super::KeccakRndRecords`]) with the words the host fill writes, at the
//! places its group's store holds them: alone, and between a wide and a packed
//! table; for the synthetic 2^16-row tables cut into an op at both ends, a
//! table cut into an op at either end, one of padding rows alone and a whole
//! padded table. A fill a row late is caught (with `test-cuda-faults`).
//!
//! Card only: without a device every upload is declined, and these tests say
//! so instead of passing.

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{TableColumns, upload_tables};
use multilinear::mle::Mle;
use multilinear::narrow::NarrowColumns;
use multilinear::records::{KeccakRndOp, Records};

use super::{KeccakRndRecords, KeccakRoundOperation};
use crate::tables::card_fill_inputs::{
    KECCAK_RND_OPS, KECCAK_RND_ROWS, KECCAK_RND_SKIP, SEEDS, keccak_ops,
};

/// Rows `skip..skip + rows` of the table `ops` make, held as their records.
fn records(ops: &[KeccakRoundOperation], skip: usize, rows: usize) -> Records {
    Records(Arc::new(KeccakRndRecords::new(
        ops.iter().map(|op| KeccakRndOp {
            timestamp: op.timestamp,
            input: op.input,
        }),
        skip,
        rows,
    )))
}

/// The tables: the synthetic sets, and three ops (72 rows) cut as the
/// finish cuts them at 32 rows a table (into the second op; from its ninth
/// row on; padding rows alone) or held whole (128 rows, 56 of padding).
fn cases() -> Vec<(String, Records)> {
    let mut cases: Vec<(String, Records)> = SEEDS
        .iter()
        .map(|&seed| {
            (
                format!("synthetic seed {seed}"),
                records(
                    &keccak_ops(seed, KECCAK_RND_OPS),
                    KECCAK_RND_SKIP,
                    KECCAK_RND_ROWS,
                ),
            )
        })
        .collect();
    let three = keccak_ops(7, 3);
    cases.push(("first cut".into(), records(&three[..2], 0, 32)));
    cases.push(("second cut".into(), records(&three[1..], 8, 32)));
    cases.push(("padding alone".into(), records(&three[3..], 24, 32)));
    cases.push(("whole".into(), records(&three, 0, 128)));
    cases
}

/// The group's store, column by column, as the card holds it.
fn uploaded(tables: &[TableColumns<'_, F>]) -> Vec<Vec<u64>> {
    upload_tables(tables)
        .expect("the card takes the group")
        .download()
        .expect("the store comes back")
}

/// Mismatched words between two stores, and the first.
fn mismatches(got: &[Vec<u64>], want: &[Vec<u64>]) -> (usize, Option<(usize, usize)>) {
    let mut count = 0;
    let mut first = None;
    for (c, (g, w)) in got.iter().zip(want).enumerate() {
        for (r, (a, b)) in g.iter().zip(w).enumerate() {
            if a != b {
                count += 1;
                first.get_or_insert((c, r));
            }
        }
    }
    (count, first)
}

fn host_words(packed: &NarrowColumns) -> Vec<Vec<u64>> {
    (0..packed.cols()).map(|c| packed.column(c)).collect()
}

#[test]
fn the_card_fills_keccak_rnd_from_its_records_as_the_host_does() {
    for (what, records) in cases() {
        let packed = records.host_fill().expect("the host fills it");
        let fills = multilinear::records::fills().0;
        let card = uploaded(&[TableColumns::Records(&records)]);
        assert_eq!(
            multilinear::records::fills().0,
            fills + 1,
            "{what}: filled on the card"
        );
        let widened = uploaded(&[TableColumns::Narrow(&packed)]);
        assert_eq!(card.len(), super::cols::NUM_COLUMNS, "{what}: the columns");
        assert!(
            card.iter().all(|c| c.len() == records.rows()),
            "{what}: the rows"
        );
        assert_eq!(
            mismatches(&card, &widened),
            (0, None),
            "{what}: against the widened host fill"
        );
        assert_eq!(
            mismatches(&card, &host_words(&packed)),
            (0, None),
            "{what}: against the host's words"
        );
    }
}

/// Between a wide table and a packed one, the filled table's columns sit
/// where its widened columns would, and the others are as uploaded.
#[test]
fn the_card_fills_keccak_rnd_in_place_among_a_groups_tables() {
    let three = keccak_ops(7, 3);
    let held = records(&three[..2], 0, 32);
    let other = records(&three[1..], 8, 32)
        .host_fill()
        .expect("the host fills it");
    let wide: Vec<Mle<F>> = (0..4u64)
        .map(|c| {
            Mle::new(
                (0..16u64)
                    .map(|r| FieldElement::from(c * 1000 + r))
                    .collect(),
            )
            .expect("a column")
        })
        .collect();
    let packed = held.host_fill().expect("the host fills it");
    let card = uploaded(&[
        TableColumns::Wide(&wide),
        TableColumns::Records(&held),
        TableColumns::Narrow(&other),
    ]);
    let host = uploaded(&[
        TableColumns::Wide(&wide),
        TableColumns::Narrow(&packed),
        TableColumns::Narrow(&other),
    ]);
    assert_eq!(card.len(), 4 + 2 * super::cols::NUM_COLUMNS);
    assert_eq!(mismatches(&card, &host), (0, None));
}

/// A fill a row late is caught word by word: the comparison above is not
/// vacuous.
#[cfg(feature = "test-cuda-faults")]
#[test]
fn a_fill_a_row_late_is_caught() {
    use std::sync::atomic::Ordering::Relaxed;
    let (what, records) = cases().swap_remove(0);
    let packed = records.host_fill().expect("the host fills it");
    math_cuda::fill::FAULT_RECORDS_SKIP.store(1, Relaxed);
    let card =
        upload_tables::<F>(&[TableColumns::Records(&records)]).and_then(|store| store.download());
    math_cuda::fill::FAULT_RECORDS_SKIP.store(0, Relaxed);
    let card = card.expect("the card takes it");
    let (count, _) = mismatches(&card, &host_words(&packed));
    assert!(count > 0, "{what}: a row late, and every word the host's");
}
