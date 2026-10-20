//! The padding of the tables a block cuts at a fixed number of rows
//! ([`CutPadding`]): the default and its knob, the count against a model of
//! the block's split, and KECCAK_RND's tables under each rule. Cut at the
//! last real row, a family's tables are the whole run's without the ones of
//! padding rows alone, and a table left out has every bus multiplicity at
//! zero.

use stark::lookup::{BusInteraction, Multiplicity};

use super::*;

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// A table as (columns, rows, words row-major), held packed or wide.
pub(super) fn shape_and_words(t: &Table) -> (usize, usize, Vec<u64>) {
    match t.narrow_main() {
        Some(packed) => {
            let columns: Vec<Vec<u64>> = (0..packed.cols()).map(|c| packed.column(c)).collect();
            let words = (0..packed.rows())
                .flat_map(|r| columns.iter().map(move |column| column[r]))
                .collect();
            (packed.cols(), packed.rows(), words)
        }
        None => {
            let main = &t.main_table;
            let words = (0..main.height)
                .flat_map(|r| (0..main.width).map(move |c| *main.get(r, c).value()))
                .collect();
            (main.width, main.height, words)
        }
    }
}

/// `Ok` when every interaction of `interactions` has multiplicity zero on
/// every row of `table`: the table adds nothing to any bus. Each multiplicity
/// must be one column, as the four cut families' are; another kind is an
/// error, never read as zero.
pub(super) fn carries_nothing(
    table: &Table,
    interactions: &[BusInteraction],
) -> Result<(), String> {
    let (cols, rows, words) = shape_and_words(table);
    for (i, interaction) in interactions.iter().enumerate() {
        let Multiplicity::Column(column) = &interaction.multiplicity else {
            return Err(format!(
                "interaction {i}: its multiplicity is not one column"
            ));
        };
        if *column >= cols {
            return Err(format!(
                "interaction {i}: multiplicity column {column} of {cols}"
            ));
        }
        if let Some(row) = (0..rows).find(|row| words[row * cols + column] != 0) {
            return Err(format!(
                "interaction {i}: multiplicity column {column} is not zero at row {row}"
            ));
        }
    }
    Ok(())
}

/// `Ok` when `kept` is `today` (a family's tables with the whole run padded)
/// without its tables of padding rows alone: `kept` is `today`'s first
/// tables word for word, every table left out carries nothing on the bus, and
/// the last table kept holds a real row.
pub(super) fn is_today_without_its_padding_tables(
    kept: &[Table],
    today: &[Table],
    interactions: &[BusInteraction],
) -> Result<(), String> {
    if kept.len() > today.len() {
        return Err(format!(
            "{} tables kept of today's {}",
            kept.len(),
            today.len()
        ));
    }
    for (k, (got, want)) in kept.iter().zip(today).enumerate() {
        if got.narrow_main().is_some() != want.narrow_main().is_some() {
            return Err(format!("table {k} is not held as today's"));
        }
        if shape_and_words(got) != shape_and_words(want) {
            return Err(format!("table {k} is not today's"));
        }
    }
    for (k, table) in today.iter().enumerate().skip(kept.len()) {
        carries_nothing(table, interactions).map_err(|e| format!("table {k}, left out: {e}"))?;
    }
    match kept.last() {
        None if today.is_empty() => Ok(()),
        None => Err("every table left out".to_string()),
        Some(last) if carries_nothing(last, interactions).is_ok() => Err(format!(
            "table {}, the last kept, is padding alone",
            kept.len() - 1
        )),
        Some(_) => Ok(()),
    }
}

/// `count` permutations with distinct states, each 24 KECCAK_RND rows.
fn round_ops(count: usize) -> Vec<KeccakRoundOperation> {
    (0..count as u64)
        .map(|i| KeccakRoundOperation {
            timestamp: 4 * i + 1,
            input: std::array::from_fn(|l| {
                (i + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((l as u64) << 7)
            }),
            output: std::array::from_fn(|l| (i + 3).wrapping_mul(0xc2b2_ae3d_27d4_eb4f) ^ l as u64),
        })
        .collect()
}

/// KECCAK_RND's tables of `rows` rows under `padding`.
fn round_tables(
    ops: &[KeccakRoundOperation],
    rows: usize,
    padding: CutPadding,
    pack: bool,
    form: TraceForm,
) -> Vec<Table> {
    keccak_rnd_chunks_in_waves(ops, rows, 0, padding, pack, form, usize::MAX).expect("the chunks")
}

/// The default stops at the last real row: the type's default, a build's
/// default settings, and what an unset variable reads as. The knob reads two
/// words (`run`, the whole run padded) and refuses the rest.
#[test]
fn the_default_is_the_last_real_row_and_the_knob_reads_two_words() {
    assert_eq!(CutPadding::default(), CutPadding::LastTable);
    assert_eq!(StreamSkip::default().cut_padding, CutPadding::LastTable);
    assert_eq!(CutPadding::parse(None), Ok(CutPadding::LastTable));
    assert_eq!(CutPadding::parse(Some(" last ")), Ok(CutPadding::LastTable));
    assert_eq!(CutPadding::parse(Some("run")), Ok(CutPadding::WholeRun));
    for other in ["", "0", "1", "Last", "whole", "last,run"] {
        let refused = CutPadding::parse(Some(other)).expect_err("another word");
        assert!(
            refused.contains("LAMBDA_VM_BLOCK_CUT_PADDING") && refused.contains("run or last"),
            "{other}: {refused}"
        );
    }
}

/// The count against a model of the block's split: the whole table is
/// `n.next_power_of_two().max(4)` rows; no taller than the cut it is one
/// table; else it is cut into pieces of `rows` rows, and piece k holds a real
/// row when its first row is one (`k·rows < n`). The whole run padded counts
/// the pieces, the last-table rule the pieces with a real row.
#[test]
fn the_count_is_the_splits_pieces_and_those_with_a_real_row() {
    for log2 in 0..=8 {
        let rows = 1usize << log2;
        for n in 0..=600usize {
            let (pieces, real) = if n == 0 {
                (0, 0)
            } else {
                let total = n.next_power_of_two().max(4);
                if total <= rows {
                    (1, 1)
                } else {
                    let pieces = total / rows;
                    (pieces, (0..pieces).filter(|k| k * rows < n).count())
                }
            };
            assert_eq!(
                CutPadding::WholeRun.tables(n, rows),
                pieces,
                "n {n} rows {rows}"
            );
            assert_eq!(
                CutPadding::LastTable.tables(n, rows),
                real,
                "n {n} rows {rows}"
            );
            assert!(real <= pieces && (real > 0) == (n > 0), "n {n} rows {rows}");
        }
    }
    // A cut of no rows is no cut: one table, never a division.
    assert_eq!(CutPadding::WholeRun.tables(9, 0), 1);
    assert_eq!(CutPadding::LastTable.tables(9, 0), 1);
}

/// The counts at the sizes of three mainnet blocks (permutations × 24
/// KECCAK_RND rows at 2^16 rows a table, ECDAS steps at 2^17), with the whole
/// run padded and cut at the last real row.
#[test]
fn the_counts_at_three_blocks_sizes() {
    for (n, rows, whole_run, last_table) in [
        (10_659 * 24, 1usize << 16, 4, 4),
        (96_384 * 24, 1 << 16, 64, 36),
        (194_333 * 24, 1 << 16, 128, 72),
        (790_528, 1 << 17, 8, 7),
    ] {
        assert_eq!(CutPadding::WholeRun.tables(n, rows), whole_run, "{n}");
        assert_eq!(CutPadding::LastTable.tables(n, rows), last_table, "{n}");
    }
}

/// KECCAK_RND with the whole run padded is the block's split of the whole
/// table; cut at the last real row it is that split without its tables of
/// padding rows alone — every op count around the cut sizes, wide, packed
/// after the build and written packed. (The split is the block prover's, a
/// host build's.)
#[cfg(feature = "parallel")]
#[test]
fn keccak_rnd_cut_at_the_last_real_row_is_the_split_without_its_padding_tables() {
    let interactions = keccak_rnd::bus_interactions();
    let mut left_out = 0usize;
    for count in [1usize, 2, 3, 5, 6, 11, 21, 22, 43] {
        let ops = round_ops(count);
        for rows in [32usize, 64, 128] {
            let what = format!("{count} permutations at {rows} rows");
            let split =
                crate::block_whir::split_rows(keccak_rnd::generate_keccak_rnd_trace(&ops), rows);
            for (pack, form) in [
                (false, TraceForm::Wide),
                (true, TraceForm::Wide),
                (true, TraceForm::Narrow),
            ] {
                let today = round_tables(&ops, rows, CutPadding::WholeRun, pack, form);
                assert_eq!(today.len(), split.len(), "{what}: the split's count");
                for (k, (got, want)) in today.iter().zip(&split).enumerate() {
                    assert_eq!(
                        shape_and_words(got),
                        shape_and_words(want),
                        "{what}: table {k} with the whole run padded"
                    );
                }
                let last = round_tables(&ops, rows, CutPadding::LastTable, pack, form);
                assert_eq!(
                    last.len(),
                    CutPadding::LastTable.tables(count * 24, rows),
                    "{what}: the count"
                );
                is_today_without_its_padding_tables(&last, &today, &interactions)
                    .unwrap_or_else(|e| panic!("{what} (pack {pack}, {form:?}): {e}"));
                left_out += today.len() - last.len();
            }
        }
    }
    assert!(left_out > 0, "no table of padding rows alone was left out");
}

/// The check the tests above lean on refuses a cut one table short of the
/// real rows (the table it leaves out holds real rounds) and a cut one table
/// long (the last table it keeps is padding alone).
#[test]
fn a_cut_one_table_short_of_the_real_rows_is_caught() {
    let interactions = keccak_rnd::bus_interactions();
    // 11 permutations = 264 rows: 512 with the run padded, eight tables of 64
    // rows, the first five with a real row.
    let ops = round_ops(11);
    let today = round_tables(&ops, 64, CutPadding::WholeRun, false, TraceForm::Wide);
    let last = round_tables(&ops, 64, CutPadding::LastTable, false, TraceForm::Wide);
    assert_eq!((today.len(), last.len()), (8, 5));
    is_today_without_its_padding_tables(&last, &today, &interactions).expect("the rule's tables");

    let short = is_today_without_its_padding_tables(&today[..4], &today, &interactions)
        .expect_err("a table with real rows left out");
    assert!(
        short.starts_with("table 4, left out:") && short.contains("is not zero at row"),
        "{short}"
    );
    let long = is_today_without_its_padding_tables(&today[..6], &today, &interactions)
        .expect_err("a table of padding rows kept");
    assert_eq!(long, "table 5, the last kept, is padding alone");
    // A multiplicity the check cannot read as a column is an error, not a zero.
    let mut unread = keccak_rnd::bus_interactions();
    unread[0].multiplicity = Multiplicity::One;
    assert!(
        carries_nothing(&today[7], &unread)
            .expect_err("a constant multiplicity")
            .contains("not one column")
    );
    carries_nothing(&today[7], &interactions).expect("a table of padding rows alone");
}
