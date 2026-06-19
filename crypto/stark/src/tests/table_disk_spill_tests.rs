use crate::table::Table;
use math::field::goldilocks::GoldilocksField;

type F = GoldilocksField;

#[test]
fn test_table_spill_roundtrip() {
    let width = 4;
    let height = 8;
    let data: Vec<math::field::element::FieldElement<F>> = (0..width * height)
        .map(|i| math::field::element::FieldElement::<F>::from(i as u64))
        .collect();

    let mut table = Table::new(data.clone(), width);
    assert!(table.mmap_backing.is_none());

    // Snapshot values before spill
    let pre_spill: Vec<Vec<math::field::element::FieldElement<F>>> = (0..height)
        .map(|r| (0..width).map(|c| *table.get(r, c)).collect())
        .collect();

    table.spill_to_disk().expect("spill_to_disk failed");
    assert!(table.mmap_backing.is_some());
    assert!(
        table.data.is_empty(),
        "heap data should be freed after spill"
    );

    // Verify get() returns the same values
    for (r, pre_row) in pre_spill.iter().enumerate() {
        for (c, pre_val) in pre_row.iter().enumerate() {
            assert_eq!(table.get(r, c), pre_val, "mismatch at ({r}, {c})");
        }
    }

    // Verify get_row() returns the same values
    for (r, pre_row) in pre_spill.iter().enumerate() {
        let row = table.get_row(r);
        assert_eq!(row.len(), width);
        for (c, pre_val) in pre_row.iter().enumerate() {
            assert_eq!(&row[c], pre_val, "get_row mismatch at ({r}, {c})");
        }
    }
}

#[test]
fn test_table_spill_empty_is_noop() {
    let mut table = Table::<F>::new(Vec::new(), 0);
    table
        .spill_to_disk()
        .expect("spill_to_disk on empty table failed");
    assert!(table.mmap_backing.is_none());
}

#[test]
fn test_table_spill_idempotent() {
    let data: Vec<math::field::element::FieldElement<F>> =
        (0..16).map(|i| math::field::element::FieldElement::<F>::from(i as u64)).collect();
    let mut table = Table::new(data, 4);

    table.spill_to_disk().expect("first spill failed");
    assert!(table.mmap_backing.is_some());

    table.spill_to_disk().expect("second spill should be no-op");
    assert!(table.mmap_backing.is_some());

    // Still readable
    assert_eq!(table.get(0, 0), &math::field::element::FieldElement::<F>::from(0u64));
    assert_eq!(table.get(3, 3), &math::field::element::FieldElement::<F>::from(15u64));
}

#[test]
fn test_clone_spilled_table_materializes_to_heap() {
    let width = 4;
    let height = 8;
    let data: Vec<math::field::element::FieldElement<F>> = (0..width * height)
        .map(|i| math::field::element::FieldElement::<F>::from(i as u64))
        .collect();

    let mut table = Table::new(data, width);
    table.spill_to_disk().expect("spill_to_disk failed");
    assert!(table.mmap_backing.is_some());

    let cloned = table.clone();
    assert!(cloned.mmap_backing.is_none(), "clone should not be spilled");
    assert_eq!(cloned.width, width);
    assert_eq!(cloned.height, height);
    assert_eq!(cloned, table, "clone must equal source element-wise");
}

#[test]
fn test_serialize_spilled_table_matches_unspilled() {
    let width = 4;
    let height = 8;
    let data: Vec<math::field::element::FieldElement<F>> = (0..width * height)
        .map(|i| math::field::element::FieldElement::<F>::from(i as u64))
        .collect();

    let unspilled = Table::new(data.clone(), width);
    let unspilled_bytes = bincode::serialize(&unspilled).expect("serialize unspilled");

    let mut spilled = Table::new(data, width);
    spilled.spill_to_disk().expect("spill_to_disk failed");
    let spilled_bytes = bincode::serialize(&spilled).expect("serialize spilled");

    assert_eq!(
        spilled_bytes, unspilled_bytes,
        "spilled and unspilled tables must serialize to identical bytes"
    );

    let restored: Table<F> =
        bincode::deserialize(&spilled_bytes).expect("deserialize spilled bytes");
    assert!(restored.mmap_backing.is_none());
    assert_eq!(restored, unspilled);
}
