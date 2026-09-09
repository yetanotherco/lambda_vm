//! Tests for BRANCH table constraints.
//!
//! These tests verify:
//! - Constraint formula correctness (IS_BIT on carries)
//! - Carry computation validity
//! - Sign extension handling

use crate::tables::branch::{BranchConstraints, BranchOperation, compute_carries};
use crate::tables::types::FE;
use stark::constraints::builder::ConstraintSet;

// =========================================================================
// Basic Constraint Property Tests
// =========================================================================

#[test]
fn test_branch_constraint_set_meta() {
    // 5 constraints: 4 conditional carry IS_BIT (degree 3) + IS_BIT<JALR> (degree 2),
    // dense and idx-ordered.
    let meta = BranchConstraints.meta();
    assert_eq!(meta.len(), 5);
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    // 4 conditional carry IS_BIT constraints are degree 3, so the table max is 3.
    assert_eq!(BranchConstraints.max_degree(), 3);
}

// =========================================================================
// Carry Computation Tests
// =========================================================================

#[test]
fn test_carry_computation_simple_positive() {
    // pc = 0x1000, offset = 4 (positive)
    // next_pc = 0x1004
    let base: u64 = 0x1000;
    let offset: u64 = 4;

    let (carry_0, carry_1) = compute_carries(base, offset);

    // Small values, no carries expected
    assert_eq!(carry_0, 0);
    assert_eq!(carry_1, 0);
}

#[test]
fn test_carry_computation_with_carry_0() {
    // pc = 0xFFFF_FFFF, offset = 1 (positive)
    // next_pc = 0x1_0000_0000
    let base: u64 = 0xFFFF_FFFF;
    let offset: u64 = 1;

    let (carry_0, carry_1) = compute_carries(base, offset);

    // Overflow from low word to high word
    assert_eq!(carry_0, 1);
    assert_eq!(carry_1, 0);
}

#[test]
fn test_carry_computation_negative_offset() {
    // pc = 0x1000, offset = -4 (sign-extended to 64-bit = 0xFFFF_FFFF_FFFF_FFFC)
    // lo: 0x1000 + 0xFFFF_FFFC = 0x1_0000_0FFC → carry_0 = 1
    // hi: 0 + 0xFFFF_FFFF + 1 = 0x1_0000_0000 → carry_1 = 1
    let base: u64 = 0x1000;
    let offset: u64 = (-4i64) as u64;

    let (carry_0, carry_1) = compute_carries(base, offset);

    assert_eq!(carry_0, 1);
    assert_eq!(carry_1, 1);
}

#[test]
fn test_carry_computation_large_negative_offset() {
    // pc = 0x1000_0000_0000, offset = -0x1000 (sign-extended = 0xFFFF_FFFF_FFFF_F000)
    // lo: 0 + 0xFFFF_F000 = 0xFFFF_F000 → carry_0 = 0
    // hi: 0x1000 + 0xFFFF_FFFF + 0 = 0x1_0000_0FFF → carry_1 = 1
    let base: u64 = 0x1000_0000_0000;
    let offset: u64 = (-0x1000i64) as u64;

    let (carry_0, carry_1) = compute_carries(base, offset);

    assert_eq!(carry_0, 0);
    assert_eq!(carry_1, 1);
}

// =========================================================================
// BranchOperation Tests
// =========================================================================

#[test]
fn test_branch_operation_simple() {
    let op = BranchOperation::new(0x1000, 4, 0, false);
    assert_eq!(op.compute_next_pc(), 0x1004);
}

#[test]
fn test_branch_operation_lsb_masking() {
    // Offset 5 should result in masked address
    let op = BranchOperation::new(0x1000, 5, 0, false);
    let next_pc = op.compute_next_pc();
    assert_eq!(next_pc, 0x1004); // 0x1005 masked to 0x1004
    assert_eq!(next_pc & 1, 0); // LSB is 0
}

#[test]
fn test_branch_operation_negative_offset() {
    // -4 sign-extended to 64-bit
    let op = BranchOperation::new(0x1000, (-4i64) as u64, 0, false);
    assert_eq!(op.compute_next_pc(), 0x0FFC);
}

#[test]
fn test_branch_operation_jalr() {
    // JALR uses register as base
    let op = BranchOperation::new(0x1000, 8, 0x2000, true);
    // Should use register (0x2000) + offset (8) = 0x2008
    assert_eq!(op.compute_next_pc(), 0x2008);
}

#[test]
fn test_branch_operation_jalr_negative_offset() {
    let op = BranchOperation::new(0x1000, (-16i64) as u64, 0x3000, true);
    // register (0x3000) + (-16) = 0x2FF0
    assert_eq!(op.compute_next_pc(), 0x2FF0);
}

#[test]
fn test_branch_operation_unmasked_vs_masked() {
    let op = BranchOperation::new(0x1000, 5, 0, false);
    let unmasked = op.compute_next_pc_unmasked();
    let masked = op.compute_next_pc();

    assert_eq!(unmasked, 0x1005);
    assert_eq!(masked, 0x1004);
    assert_eq!(masked, unmasked & !1);
}

// =========================================================================
// Sign Extension Tests
// =========================================================================

#[test]
fn test_sign_extension_positive() {
    // Offset with MSB=0 (positive): no sign extension
    let offset: u32 = 0x7FFF_FFFF; // Max positive 32-bit signed
    let extended = offset as i32 as i64 as u64;
    assert_eq!(extended >> 32, 0); // High word should be 0
}

#[test]
fn test_sign_extension_negative() {
    // Offset with MSB=1 (negative): sign extension fills high word
    let offset: u32 = 0x8000_0000; // Min negative 32-bit signed
    let extended = offset as i32 as i64 as u64;
    assert_eq!(extended >> 32, 0xFFFF_FFFF); // High word should be all 1s
}

#[test]
fn test_sign_extension_minus_one() {
    // -1 in 32-bit is 0xFFFFFFFF
    let offset: u32 = 0xFFFF_FFFF;
    let extended = offset as i32 as i64 as u64;
    assert_eq!(extended, 0xFFFF_FFFF_FFFF_FFFF); // Full 64-bit -1
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_branch_zero_values() {
    let op = BranchOperation::new(0, 0, 0, false);
    assert_eq!(op.compute_next_pc(), 0);
}

#[test]
fn test_branch_max_pc_small_offset() {
    let op = BranchOperation::new(0xFFFF_FFFF_FFFF_FFF0, 4, 0, false);
    // Should wrap around
    let next_pc = op.compute_next_pc();
    assert_eq!(next_pc, 0xFFFF_FFFF_FFFF_FFF4);
}

#[test]
fn test_branch_overflow_wraps() {
    // PC at max, positive offset causes wrap
    let op = BranchOperation::new(0xFFFF_FFFF_FFFF_FFFE, 4, 0, false);
    let next_pc = op.compute_next_pc();
    // Wraps around: 0xFFFF_FFFF_FFFF_FFFE + 4 = 0x2 (mod 2^64), masked to 0x2
    assert_eq!(next_pc, 0x2);
}

// =========================================================================
// IS_BIT Formula Tests
// =========================================================================

#[test]
fn test_is_bit_formula_valid_zero() {
    // IS_BIT formula: X * (1 - X) = 0
    // When X = 0: 0 * 1 = 0 ✓
    let x = FE::zero();
    let result = x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_valid_one() {
    // IS_BIT formula: X * (1 - X) = 0
    // When X = 1: 1 * 0 = 0 ✓
    let x = FE::one();
    let result = x * (FE::one() - x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_invalid_two() {
    // IS_BIT formula: X * (1 - X) = 0
    // When X = 2: 2 * (-1) = -2 ≠ 0 ✗
    let x = FE::from(2u64);
    let result = x * (FE::one() - x);
    assert_ne!(result, FE::zero());
}
