//! Unit tests for BusValue and LinearTerm combine logic.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::lookup::{BusValue, LinearTerm, Packing};

type F = GoldilocksField;
type FE = FieldElement<F>;

// =============================================================================
// BusValue::combine_from tests
// =============================================================================

#[test]
fn test_packed_delegates_to_packing() {
    // BusValue::Packed should produce same result as Packing::combine
    let bv = BusValue::Packed {
        start_column: 0,
        packing: Packing::Word2L,
    };

    // Simulate columns: [0x1234, 0x5678]
    let columns = [FE::from(0x1234u64), FE::from(0x5678u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    // Expected: 0x56781234 (same as Packing::Word2L)
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(0x56781234u64));
}

#[test]
fn test_linear_single_column() {
    // Linear with single column: 1*col[0]
    let bv = BusValue::Linear(vec![LinearTerm::Column {
        coefficient: 1,
        column: 0,
    }]);

    let columns = [FE::from(42u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(42u64));
}

#[test]
fn test_linear_column_with_coefficient() {
    // Linear with coefficient: 5*col[0]
    let bv = BusValue::Linear(vec![LinearTerm::Column {
        coefficient: 5,
        column: 0,
    }]);

    let columns = [FE::from(10u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(50u64)); // 5 * 10 = 50
}

#[test]
fn test_linear_multiple_columns() {
    // Linear with multiple columns: 3*col[0] + 7*col[1]
    let bv = BusValue::Linear(vec![
        LinearTerm::Column {
            coefficient: 3,
            column: 0,
        },
        LinearTerm::Column {
            coefficient: 7,
            column: 1,
        },
    ]);

    let columns = [FE::from(10u64), FE::from(20u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(170u64)); // 3*10 + 7*20 = 30 + 140 = 170
}

#[test]
fn test_linear_constant_only() {
    // Linear with constant only
    let bv = BusValue::Linear(vec![LinearTerm::Constant(42)]);

    // No columns needed
    let combined: Vec<FE> = bv.combine_from(|_| panic!("should not access columns"));

    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(42u64));
}

#[test]
fn test_linear_mixed_columns_and_constant() {
    // Linear mixed: 2*col[0] + 3*col[1] + 10
    let bv = BusValue::Linear(vec![
        LinearTerm::Column {
            coefficient: 2,
            column: 0,
        },
        LinearTerm::Column {
            coefficient: 3,
            column: 1,
        },
        LinearTerm::Constant(10),
    ]);

    let columns = [FE::from(5u64), FE::from(7u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(41u64)); // 2*5 + 3*7 + 10 = 10 + 21 + 10 = 41
}

// =============================================================================
// Helper function tests
// =============================================================================

#[test]
fn test_constant_helper() {
    // BusValue::constant(42) should create Linear([Constant(42)])
    let bv = BusValue::constant(42);

    match bv {
        BusValue::Linear(terms) => {
            assert_eq!(terms.len(), 1);
            match &terms[0] {
                LinearTerm::Constant(v) => assert_eq!(*v, 42),
                _ => panic!("expected Constant term"),
            }
        }
        _ => panic!("expected Linear variant"),
    }
}

#[test]
fn test_column_helper() {
    // BusValue::column(3) should create Linear([Column{coeff:1, col:3}])
    let bv = BusValue::column(3);

    match bv {
        BusValue::Linear(terms) => {
            assert_eq!(terms.len(), 1);
            match &terms[0] {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1);
                    assert_eq!(*column, 3);
                }
                _ => panic!("expected Column term"),
            }
        }
        _ => panic!("expected Linear variant"),
    }
}

#[test]
fn test_linear_helper() {
    // BusValue::linear(terms) should wrap terms correctly
    let terms = vec![
        LinearTerm::Column {
            coefficient: 2,
            column: 0,
        },
        LinearTerm::Constant(5),
    ];
    let bv = BusValue::linear(terms);

    match bv {
        BusValue::Linear(t) => {
            assert_eq!(t.len(), 2);
        }
        _ => panic!("expected Linear variant"),
    }
}

// =============================================================================
// Metadata tests
// =============================================================================

#[test]
fn test_linear_num_bus_elements() {
    // Linear always produces 1 bus element
    let bv1 = BusValue::constant(42);
    let bv2 = BusValue::Linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: 0,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: 1,
        },
        LinearTerm::Constant(10),
    ]);

    assert_eq!(bv1.num_bus_elements(), 1);
    assert_eq!(bv2.num_bus_elements(), 1);
}

#[test]
fn test_packed_num_bus_elements() {
    // Packed delegates to Packing::num_bus_elements
    let bv_direct = BusValue::Packed {
        start_column: 0,
        packing: Packing::Direct,
    };
    let bv_word2l = BusValue::Packed {
        start_column: 0,
        packing: Packing::Word2L,
    };
    let bv_dwordhl = BusValue::Packed {
        start_column: 0,
        packing: Packing::DWordHL,
    };

    assert_eq!(bv_direct.num_bus_elements(), 1);
    assert_eq!(bv_word2l.num_bus_elements(), 1);
    assert_eq!(bv_dwordhl.num_bus_elements(), 2); // DWordHL produces 2 elements
}

#[test]
fn test_packed_column_indices() {
    // Packed returns range [start_column, start_column + num_columns)
    let bv = BusValue::Packed {
        start_column: 3,
        packing: Packing::Word2L, // 2 columns
    };

    let indices = bv.column_indices();
    assert_eq!(indices, vec![3, 4]);
}

#[test]
fn test_linear_column_indices() {
    // Linear returns only columns from Column terms (not constants)
    let bv = BusValue::Linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: 2,
        },
        LinearTerm::Constant(42), // should not appear in indices
        LinearTerm::Column {
            coefficient: 3,
            column: 5,
        },
    ]);

    let indices = bv.column_indices();
    assert_eq!(indices, vec![2, 5]);
}

#[test]
fn test_constant_column_indices_empty() {
    // Constant-only Linear should have empty column_indices
    let bv = BusValue::constant(42);

    let indices = bv.column_indices();
    assert!(indices.is_empty());
}

// =============================================================================
// LinearTerm::ColumnUnsigned tests
// =============================================================================

#[test]
fn test_linear_column_unsigned_single() {
    // Linear with single unsigned column: large_coeff * col[0]
    let large_coeff: u64 = 0x8000_0000_0000_0000; // Larger than i64::MAX
    let bv = BusValue::Linear(vec![LinearTerm::ColumnUnsigned {
        coefficient: large_coeff,
        column: 0,
    }]);

    let columns = [FE::from(2u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    // Result = large_coeff * 2 in field
    let expected = FE::from(large_coeff) * FE::from(2u64);
    assert_eq!(combined[0], expected);
}

#[test]
fn test_linear_column_unsigned_with_coefficient() {
    // Linear with unsigned coefficient that fits in u64 but not i64
    let coeff: u64 = 18446744065119617026; // Example: INV_2_32 in Goldilocks
    let bv = BusValue::Linear(vec![LinearTerm::ColumnUnsigned {
        coefficient: coeff,
        column: 0,
    }]);

    let columns = [FE::from(100u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    let expected = FE::from(coeff) * FE::from(100u64);
    assert_eq!(combined[0], expected);
}

#[test]
fn test_linear_mixed_column_and_column_unsigned() {
    // Mix of Column (signed) and ColumnUnsigned: 3*col[0] + large*col[1]
    let large_coeff: u64 = 0x8000_0000_0000_0000;
    let bv = BusValue::Linear(vec![
        LinearTerm::Column {
            coefficient: 3,
            column: 0,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: large_coeff,
            column: 1,
        },
    ]);

    let columns = [FE::from(10u64), FE::from(2u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    // 3*10 + large_coeff*2
    let expected = FE::from(30u64) + FE::from(large_coeff) * FE::from(2u64);
    assert_eq!(combined[0], expected);
}

#[test]
fn test_linear_mixed_all_term_types() {
    // Mix of all term types: 2*col[0] + large*col[1] + 10
    let large_coeff: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    let bv = BusValue::Linear(vec![
        LinearTerm::Column {
            coefficient: 2,
            column: 0,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: large_coeff,
            column: 1,
        },
        LinearTerm::Constant(10),
    ]);

    let columns = [FE::from(5u64), FE::from(3u64)];
    let combined: Vec<FE> = bv.combine_from(|col| columns[col].clone());

    assert_eq!(combined.len(), 1);
    // 2*5 + large*3 + 10
    let expected = FE::from(10u64) + FE::from(large_coeff) * FE::from(3u64) + FE::from(10u64);
    assert_eq!(combined[0], expected);
}

#[test]
fn test_linear_column_unsigned_column_indices() {
    // ColumnUnsigned should appear in column_indices
    let bv = BusValue::Linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: 2,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: 0x8000_0000_0000_0000,
            column: 7,
        },
        LinearTerm::Constant(42),
    ]);

    let indices = bv.column_indices();
    assert_eq!(indices, vec![2, 7]);
}

#[test]
fn test_linear_column_unsigned_only_indices() {
    // Only ColumnUnsigned terms
    let bv = BusValue::Linear(vec![
        LinearTerm::ColumnUnsigned {
            coefficient: 100,
            column: 3,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: 200,
            column: 8,
        },
    ]);

    let indices = bv.column_indices();
    assert_eq!(indices, vec![3, 8]);
}
