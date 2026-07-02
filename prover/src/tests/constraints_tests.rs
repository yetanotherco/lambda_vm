//! Tests for the 64-bit VM constraint templates.

use crate::constraints::templates::{AddLinearTerm, AddOperand, SHIFT_32};
use crate::tables::types::FE;

// =========================================================================
// Basic tests
// =========================================================================

#[test]
fn test_inv_2_32() {
    // Verify that 2^32 * 2^(-32) = 1 in Goldilocks
    let two_32 = FE::from(SHIFT_32);
    let inv = two_32.inv().expect("Should be invertible");
    let product = two_32 * inv;
    assert_eq!(product, FE::one());
}

// =========================================================================
// IS_BIT formula verification tests
// =========================================================================

#[test]
fn test_is_bit_formula_valid_zero() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 0: cond * 0 * 1 = 0 ✓
    let cond = FE::one();
    let x = FE::zero();
    let result = cond * x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_valid_one() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 1: cond * 1 * 0 = 0 ✓
    let cond = FE::one();
    let x = FE::one();
    let result = cond * x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_invalid_two() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 2: cond * 2 * (-1) = -2 ≠ 0 ✗
    let cond = FE::one();
    let x = FE::from(2u64);
    let result = cond * x * (FE::one() - x);
    assert_ne!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_cond_zero() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When cond = 0: 0 * X * (1 - X) = 0 always ✓
    let cond = FE::zero();
    let x = FE::from(42u64); // Any invalid value
    let result = cond * x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

// =========================================================================
// ADD carry computation verification tests
// =========================================================================

#[test]
fn test_carry_computation_no_carry() {
    // lhs_lo + rhs_lo < 2^32, so carry_0 = 0
    let lhs_lo = FE::from(100u64);
    let rhs_lo = FE::from(200u64);
    let sum_lo = FE::from(300u64); // 100 + 200 = 300

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    // carry should be 0
    assert_eq!(carry, FE::zero());
}

#[test]
fn test_carry_computation_with_carry() {
    // lhs_lo + rhs_lo >= 2^32, so carry_0 = 1
    let lhs_lo = FE::from(0xFFFFFFFFu64); // 2^32 - 1
    let rhs_lo = FE::from(2u64);
    // sum_lo = (0xFFFFFFFF + 2) mod 2^32 = 1
    let sum_lo = FE::from(1u64);

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    // carry should be 1: (0xFFFFFFFF + 2 - 1) / 2^32 = 0x100000000 / 2^32 = 1
    assert_eq!(carry, FE::one());
}

#[test]
fn test_carry_is_bit_valid() {
    // When carry is 0, the IS_BIT constraint is satisfied
    let carry = FE::zero();
    let result = carry * (FE::one() - carry);
    assert_eq!(result, FE::zero());

    // When carry is 1, the IS_BIT constraint is satisfied
    let carry = FE::one();
    let result = carry * (FE::one() - carry);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_carry_boundary_just_below() {
    // lhs_lo + rhs_lo = 2^32 - 1 (no carry)
    let lhs_lo = FE::from(0x80000000u64); // 2^31
    let rhs_lo = FE::from(0x7FFFFFFFu64); // 2^31 - 1
    let sum_lo = FE::from(0xFFFFFFFFu64); // 2^32 - 1

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    assert_eq!(carry, FE::zero());
}

#[test]
fn test_carry_boundary_exactly_2_32() {
    // lhs_lo + rhs_lo = 2^32 (carry = 1)
    let lhs_lo = FE::from(0x80000000u64); // 2^31
    let rhs_lo = FE::from(0x80000000u64); // 2^31
    let sum_lo = FE::from(0u64); // (2^31 + 2^31) mod 2^32 = 0

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    assert_eq!(carry, FE::one());
}

#[test]
fn test_carry_max_values() {
    // lhs_lo = 0xFFFFFFFF, rhs_lo = 0xFFFFFFFF
    // sum = 0x1FFFFFFFE, sum_lo = 0xFFFFFFFE, carry = 1
    let lhs_lo = FE::from(0xFFFFFFFFu64);
    let rhs_lo = FE::from(0xFFFFFFFFu64);
    let sum_lo = FE::from(0xFFFFFFFEu64);

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (lhs_lo + rhs_lo - sum_lo) * inv_2_32;

    assert_eq!(carry, FE::one());
}

// =========================================================================
// AddOperand tests
// =========================================================================

#[test]
fn test_add_operand_constant() {
    // Test that constant operand creates correct Linear representation
    let op = AddOperand::constant(42);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 1);
            assert!(hi.is_empty());
            match &lo[0] {
                AddLinearTerm::Constant(v) => assert_eq!(*v, 42),
                _ => panic!("Expected Constant"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_from_word() {
    // Test Word → DWordWL (single column, zero-extended)
    let op = AddOperand::from_word(5);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 1);
            assert!(hi.is_empty());
            match &lo[0] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1);
                    assert_eq!(*column, 5);
                }
                _ => panic!("Expected Column"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_dword() {
    // Test DWordWL (2 consecutive columns)
    let op = AddOperand::dword(10);
    match op {
        AddOperand::DWordWL { start_column } => {
            assert_eq!(start_column, 10);
        }
        _ => panic!("Expected DWordWL variant"),
    }
}

#[test]
fn test_add_operand_from_dword_hl() {
    // Test DWordHL → DWordWL (repack 4 halves → 2 words)
    let op = AddOperand::from_dword_hl(0);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 2);
            assert_eq!(hi.len(), 2);

            // lo = h[0] + 2^16 * h[1]
            match &lo[0] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1);
                    assert_eq!(*column, 0);
                }
                _ => panic!("Expected Column"),
            }
            match &lo[1] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1 << 16);
                    assert_eq!(*column, 1);
                }
                _ => panic!("Expected Column"),
            }

            // hi = h[2] + 2^16 * h[3]
            match &hi[0] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1);
                    assert_eq!(*column, 2);
                }
                _ => panic!("Expected Column"),
            }
            match &hi[1] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, 1 << 16);
                    assert_eq!(*column, 3);
                }
                _ => panic!("Expected Column"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_from_dword_bl() {
    // Test DWordBL → DWordWL (repack 8 bytes → 2 words)
    let op = AddOperand::from_dword_bl(0);
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 4);
            assert_eq!(hi.len(), 4);

            // Check lo coefficients: 1, 2^8, 2^16, 2^24
            let lo_coeffs: Vec<i64> = lo
                .iter()
                .map(|t| match t {
                    AddLinearTerm::Column { coefficient, .. } => *coefficient,
                    _ => panic!("Expected Column"),
                })
                .collect();
            assert_eq!(lo_coeffs, vec![1, 1 << 8, 1 << 16, 1 << 24]);

            // Check hi coefficients: 1, 2^8, 2^16, 2^24
            let hi_coeffs: Vec<i64> = hi
                .iter()
                .map(|t| match t {
                    AddLinearTerm::Column { coefficient, .. } => *coefficient,
                    _ => panic!("Expected Column"),
                })
                .collect();
            assert_eq!(hi_coeffs, vec![1, 1 << 8, 1 << 16, 1 << 24]);

            // Check hi columns: 4, 5, 6, 7
            let hi_cols: Vec<usize> = hi
                .iter()
                .map(|t| match t {
                    AddLinearTerm::Column { column, .. } => *column,
                    _ => panic!("Expected Column"),
                })
                .collect();
            assert_eq!(hi_cols, vec![4, 5, 6, 7]);
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_linear_with_negative_coefficient() {
    // Test linear operand with negative coefficient: 4 - 2*c
    // This represents expressions like `4 - 2 * c_type_instruction`
    let op = AddOperand::linear(
        vec![
            AddLinearTerm::Constant(4),
            AddLinearTerm::Column {
                coefficient: -2,
                column: 0,
            },
        ],
        vec![], // hi = 0
    );
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 2);
            assert!(hi.is_empty());

            match &lo[0] {
                AddLinearTerm::Constant(v) => assert_eq!(*v, 4),
                _ => panic!("Expected Constant"),
            }
            match &lo[1] {
                AddLinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    assert_eq!(*coefficient, -2);
                    assert_eq!(*column, 0);
                }
                _ => panic!("Expected Column"),
            }
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_add_operand_linear_with_nonzero_hi() {
    // Test linear operand with non-trivial hi terms (virtual column case)
    let op = AddOperand::linear(
        vec![
            AddLinearTerm::Column {
                coefficient: 1 << 16,
                column: 0,
            },
            AddLinearTerm::Column {
                coefficient: 1 << 8,
                column: 1,
            },
            AddLinearTerm::Column {
                coefficient: 1,
                column: 2,
            },
        ],
        vec![
            AddLinearTerm::Column {
                coefficient: 1 << 16,
                column: 3,
            },
            AddLinearTerm::Column {
                coefficient: 1,
                column: 4,
            },
        ],
    );
    match op {
        AddOperand::Linear { lo, hi } => {
            assert_eq!(lo.len(), 3);
            assert_eq!(hi.len(), 2);
        }
        _ => panic!("Expected Linear variant"),
    }
}

#[test]
fn test_linear_term_negative_coefficient_formula() {
    // Verify that 4 - 2*1 = 2 using field arithmetic
    // This simulates: constant(4) + coefficient(-2) * column_value(1)
    let constant = FE::from(4u64);
    let coefficient = -FE::from(2u64);
    let col_value = FE::one();

    let result = constant + coefficient * col_value;
    assert_eq!(result, FE::from(2u64));
}

#[test]
fn test_dword_hl_repack_formula() {
    // Verify DWordHL → DWordWL repacking formula
    // lo = h[0] + 2^16 * h[1]
    // hi = h[2] + 2^16 * h[3]
    //
    // Example: h = [0x1234, 0x5678, 0xABCD, 0xEF01]
    // lo = 0x1234 + 0x5678_0000 = 0x5678_1234
    // hi = 0xABCD + 0xEF01_0000 = 0xEF01_ABCD

    let h0 = FE::from(0x1234u64);
    let h1 = FE::from(0x5678u64);
    let h2 = FE::from(0xABCDu64);
    let h3 = FE::from(0xEF01u64);

    let shift_16 = FE::from(1u64 << 16);

    let lo = h0 + h1 * shift_16;
    let hi = h2 + h3 * shift_16;

    assert_eq!(lo, FE::from(0x5678_1234u64));
    assert_eq!(hi, FE::from(0xEF01_ABCDu64));
}

#[test]
fn test_dword_bl_repack_formula() {
    // Verify DWordBL → DWordWL repacking formula
    // lo = b[0] + 2^8*b[1] + 2^16*b[2] + 2^24*b[3]
    // hi = b[4] + 2^8*b[5] + 2^16*b[6] + 2^24*b[7]
    //
    // Example: bytes = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]
    // lo = 0x78563412
    // hi = 0xF0DEBC9A

    let b: Vec<FE> = vec![
        FE::from(0x12u64),
        FE::from(0x34u64),
        FE::from(0x56u64),
        FE::from(0x78u64),
        FE::from(0x9Au64),
        FE::from(0xBCu64),
        FE::from(0xDEu64),
        FE::from(0xF0u64),
    ];

    let coeffs: Vec<FE> = vec![
        FE::from(1u64),
        FE::from(1u64 << 8),
        FE::from(1u64 << 16),
        FE::from(1u64 << 24),
    ];

    let lo = b[0] * coeffs[0] + b[1] * coeffs[1] + b[2] * coeffs[2] + b[3] * coeffs[3];
    let hi = b[4] * coeffs[0] + b[5] * coeffs[1] + b[6] * coeffs[2] + b[7] * coeffs[3];

    assert_eq!(lo, FE::from(0x78563412u64));
    assert_eq!(hi, FE::from(0xF0DEBC9Au64));
}

// =========================================================================
// CPU Constraints tests
// =========================================================================

use crate::constraints::cpu::{BIT_FLAG_COLUMNS, CpuConstraints, NUM_CPU_CONSTRAINTS};
use crate::tables::cpu::cols as cpu_cols;
use stark::constraints::builder::{ConstraintSet, num_base_from_meta};

#[test]
fn test_cpu_bit_flag_columns_count() {
    // 10 top-level flags + pc_double_read + prev_pc_timestamp_borrow + non_padding.
    assert_eq!(BIT_FLAG_COLUMNS.len(), 12);
}

#[test]
fn test_cpu_bit_flag_columns_valid() {
    for &col in BIT_FLAG_COLUMNS {
        assert!(col < cpu_cols::NUM_COLUMNS, "Column {} out of range", col);
    }
}

#[test]
fn test_cpu_constraint_set_meta_is_dense_all_base() {
    // The CPU single-source set declares exactly NUM_CPU_CONSTRAINTS base
    // constraints, dense and idx-ordered (per-constraint degrees and the
    // folder-vs-capture faithfulness are covered by constraint_set_tests_b).
    let meta = CpuConstraints.meta();
    assert_eq!(meta.len(), NUM_CPU_CONSTRAINTS);
    assert_eq!(num_base_from_meta(&meta), NUM_CPU_CONSTRAINTS);
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(
            m.constraint_idx, i,
            "constraint indices cover 0..N in order"
        );
    }
}
