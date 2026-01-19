//! Tests for the 64-bit VM constraint templates.

use crate::constraints64::templates::{
    AddConstraint, IsBitConstraint, SHIFT_32, new_is_bit_constraints,
};
use crate::tables64::types::FE;
use stark::constraints::transition::TransitionConstraint;

// =========================================================================
// Basic tests
// =========================================================================

#[test]
fn test_inv_2_32() {
    // Verify that 2^32 * 2^(-32) = 1 in Goldilocks
    let two_32 = FE::from(SHIFT_32);
    let inv = two_32.inv().expect("Should be invertible");
    let product = &two_32 * &inv;
    assert_eq!(product, FE::one());
}

#[test]
fn test_is_bit_constraint_degree() {
    let constraint = IsBitConstraint::new(0, 1, 0);
    assert_eq!(constraint.degree(), 3);
}

#[test]
fn test_add_constraint_degree() {
    let (c0, c1) = AddConstraint::new_pair(0, 1, 3, 5, 0);
    assert_eq!(c0.degree(), 3);
    assert_eq!(c1.degree(), 3);
}

#[test]
fn test_add_constraint_indices() {
    let (c0, c1) = AddConstraint::new_pair(0, 1, 3, 5, 10);
    assert_eq!(c0.constraint_idx(), 10);
    assert_eq!(c1.constraint_idx(), 11);
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
    let result = &cond * &x * (FE::one() - &x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_valid_one() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 1: cond * 1 * 0 = 0 ✓
    let cond = FE::one();
    let x = FE::one();
    let result = &cond * &x * (FE::one() - &x);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_invalid_two() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When X = 2: cond * 2 * (-1) = -2 ≠ 0 ✗
    let cond = FE::one();
    let x = FE::from(2u64);
    let result = &cond * &x * (FE::one() - &x);
    assert_ne!(result, FE::zero());
}

#[test]
fn test_is_bit_formula_cond_zero() {
    // IS_BIT formula: cond * X * (1 - X) = 0
    // When cond = 0: 0 * X * (1 - X) = 0 always ✓
    let cond = FE::zero();
    let x = FE::from(42u64); // Any invalid value
    let result = &cond * &x * (FE::one() - &x);
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
    let carry = (&lhs_lo + &rhs_lo - &sum_lo) * &inv_2_32;

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
    let carry = (&lhs_lo + &rhs_lo - &sum_lo) * &inv_2_32;

    // carry should be 1: (0xFFFFFFFF + 2 - 1) / 2^32 = 0x100000000 / 2^32 = 1
    assert_eq!(carry, FE::one());
}

#[test]
fn test_carry_is_bit_valid() {
    // When carry is 0, the IS_BIT constraint is satisfied
    let carry = FE::zero();
    let result = &carry * (FE::one() - &carry);
    assert_eq!(result, FE::zero());

    // When carry is 1, the IS_BIT constraint is satisfied
    let carry = FE::one();
    let result = &carry * (FE::one() - &carry);
    assert_eq!(result, FE::zero());
}

#[test]
fn test_carry_boundary_just_below() {
    // lhs_lo + rhs_lo = 2^32 - 1 (no carry)
    let lhs_lo = FE::from(0x80000000u64); // 2^31
    let rhs_lo = FE::from(0x7FFFFFFFu64); // 2^31 - 1
    let sum_lo = FE::from(0xFFFFFFFFu64); // 2^32 - 1

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (&lhs_lo + &rhs_lo - &sum_lo) * &inv_2_32;

    assert_eq!(carry, FE::zero());
}

#[test]
fn test_carry_boundary_exactly_2_32() {
    // lhs_lo + rhs_lo = 2^32 (carry = 1)
    let lhs_lo = FE::from(0x80000000u64); // 2^31
    let rhs_lo = FE::from(0x80000000u64); // 2^31
    let sum_lo = FE::from(0u64); // (2^31 + 2^31) mod 2^32 = 0

    let inv_2_32 = FE::from(SHIFT_32).inv().unwrap();
    let carry = (&lhs_lo + &rhs_lo - &sum_lo) * &inv_2_32;

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
    let carry = (&lhs_lo + &rhs_lo - &sum_lo) * &inv_2_32;

    assert_eq!(carry, FE::one());
}

// =========================================================================
// Helper function tests
// =========================================================================

#[test]
fn test_new_is_bit_constraints_count() {
    let (constraints, next_idx) = new_is_bit_constraints(0, &[1, 2, 3, 4], 10);
    assert_eq!(constraints.len(), 4);
    assert_eq!(next_idx, 14);
}

#[test]
fn test_new_is_bit_constraints_indices() {
    let (constraints, _) = new_is_bit_constraints(0, &[5, 6, 7], 100);
    assert_eq!(constraints[0].constraint_idx(), 100);
    assert_eq!(constraints[1].constraint_idx(), 101);
    assert_eq!(constraints[2].constraint_idx(), 102);
}
