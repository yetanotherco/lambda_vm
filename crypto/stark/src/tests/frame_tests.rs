use crate::frame::{Frame, RowFrame};
use crate::trace::LDETraceTable;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;

type Fp = FieldElement<Gl>;
type Fp3 = FieldElement<Ext3>;

/// An 8-row, 2-main/1-aux LDE table (blowup 2) with distinct per-cell
/// values, so any mis-indexed read is caught by value.
fn table() -> LDETraceTable<Gl, Ext3> {
    let main: Vec<Vec<Fp>> = (0..2)
        .map(|c| (0..8).map(|r| Fp::from((100 * c + r) as u64)).collect())
        .collect();
    let aux: Vec<Vec<Fp3>> = vec![
        (0..8)
            .map(|r| Fp3::new([Fp::from(1000 + r as u64), Fp::zero(), Fp::zero()]))
            .collect(),
    ];
    LDETraceTable::from_columns(main, aux, 1, 2)
}

#[test]
fn borrows_rows_at_each_offset() {
    let t = table();
    let rows = RowFrame::from_lde(&t, 3, &[0, 1]);
    // offset 0 -> row 3; offset 1 -> row 3 + lde_step_size (= blowup 2) = 5.
    assert_eq!(rows.main(0, 0), t.get_main(3, 0));
    assert_eq!(rows.main(0, 1), t.get_main(3, 1));
    assert_eq!(rows.main(1, 0), t.get_main(5, 0));
    assert_eq!(rows.aux(0, 0), t.get_aux(3, 0));
    assert_eq!(rows.aux(1, 0), t.get_aux(5, 0));
    assert_eq!(rows.num_offsets(), 2);
}

#[test]
fn wraps_cyclically_at_the_domain_end() {
    let t = table();
    // Last LDE row: offset 1 reads (7 + 2) % 8 = row 1.
    let rows = RowFrame::from_lde(&t, 7, &[0, 1]);
    assert_eq!(rows.main(0, 0), t.get_main(7, 0));
    assert_eq!(rows.main(1, 0), t.get_main(1, 0));
    assert_eq!(rows.aux(1, 0), t.get_aux(1, 0));
}

#[test]
#[should_panic(expected = "at most")]
fn rejects_too_many_offsets() {
    let t = table();
    let _ = RowFrame::from_lde(&t, 0, &[0, 1, 2, 3, 4]);
}

#[test]
fn as_row_frame_matches_owned_frame() {
    let t = table();
    let frame = Frame::read_step_from_lde(&t, 2, &[0, 1]);
    let rows = frame.as_row_frame();
    let direct = RowFrame::from_lde(&t, t.step_to_row(2), &[0, 1]);
    for offset in 0..2 {
        for col in 0..2 {
            assert_eq!(rows.main(offset, col), direct.main(offset, col));
        }
        assert_eq!(rows.aux(offset, 0), direct.aux(offset, 0));
    }
}
