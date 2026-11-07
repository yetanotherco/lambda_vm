//! `Table::columns_blocked` is `Table::columns`, tile by tile.

use crate::table::Table;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

type FE = FieldElement<GoldilocksField>;

fn table(width: usize, height: usize) -> Table<GoldilocksField> {
    let data: Vec<FE> = (0..width * height)
        .map(|i| FE::from((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7))
        .collect();
    Table::new(data, width)
}

/// Every shape a block has — one column, fewer columns than a band, bands and a
/// remainder, a KECCAK_RND-wide table — at heights below, at and past a row
/// block, including a partial last block.
#[test]
fn the_blocked_transposition_is_the_transposition() {
    for width in [1, 3, 64, 70, 1480] {
        for height in [1, 4, 1024, 1500, 4096] {
            let t = table(width, height);
            assert_eq!(t.columns_blocked(), t.columns(), "{width} x {height}");
        }
    }
}

#[test]
fn an_empty_table_has_no_columns() {
    let t = Table::<GoldilocksField>::new(Vec::new(), 0);
    assert!(t.columns_blocked().is_empty());
}
