//! Helper functions for common constraint patterns used with AirBuilder.
//!
//! These helpers provide common constraint patterns that are frequently used
//! across multiple tables, reducing code duplication and improving readability.
//!
//! Two sets of helpers are provided:
//! - Extension-field versions (`assert_is_bit`, etc.) for `AirBuilder<F>` (E*E, verifier + prover)
//! - Base-field versions (`assert_is_bit_base`, etc.) for `MainAirBuilder<F, E>` (F*E, 3 base muls, prover only)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::air_builder::{AirBuilder, MainAirBuilder};

// =============================================================================
// Extension-field helpers (AirBuilder<F>) - used by verifier + prover fallback
// =============================================================================

/// IS_BIT: x*(1-x) == 0
///
/// Enforces that `x` is binary (0 or 1).
/// Degree: 2
#[inline]
pub fn assert_is_bit<F: IsField>(builder: &mut dyn AirBuilder<F>, x: FieldElement<F>) {
    let one = FieldElement::<F>::one();
    builder.assert_zero(x.clone() * (one - x));
}

/// Conditional IS_BIT: cond * x * (1-x) == 0
///
/// Enforces that `x` is binary only when `cond` is 1.
/// When `cond` is 0, the constraint is automatically satisfied.
/// Degree: 3
#[inline]
pub fn assert_is_bit_cond<F: IsField>(
    builder: &mut dyn AirBuilder<F>,
    cond: FieldElement<F>,
    x: FieldElement<F>,
) {
    let one = FieldElement::<F>::one();
    builder.assert_zero(cond * x.clone() * (one - x));
}

/// Zero when flag is off: (1 - flag) * value == 0
///
/// Enforces that `value` is zero when `flag` is 0.
/// When `flag` is 1, the constraint is automatically satisfied.
/// Degree: 2
#[inline]
pub fn assert_zero_when_off<F: IsField>(
    builder: &mut dyn AirBuilder<F>,
    flag: FieldElement<F>,
    value: FieldElement<F>,
) {
    builder.assert_zero((FieldElement::<F>::one() - flag) * value);
}

/// Zero when flag is on: flag * value == 0
///
/// Enforces that `value` is zero when `flag` is 1.
/// When `flag` is 0, the constraint is automatically satisfied.
/// Degree: 2
#[inline]
pub fn assert_zero_when_on<F: IsField>(
    builder: &mut dyn AirBuilder<F>,
    flag: FieldElement<F>,
    value: FieldElement<F>,
) {
    builder.assert_zero(flag * value);
}

/// Equality constraint: (1 - flag) * (left - right) == 0
///
/// Enforces that `left == right` when `flag` is 1.
/// When `flag` is 0, the constraint is automatically satisfied.
/// Degree: 2
#[inline]
pub fn assert_eq_when_on<F: IsField>(
    builder: &mut dyn AirBuilder<F>,
    flag: FieldElement<F>,
    left: FieldElement<F>,
    right: FieldElement<F>,
) {
    builder.assert_zero(flag * (left - right));
}

/// Inequality constraint: flag * (left - right) == 0
///
/// Enforces that `left == right` when `flag` is 0.
/// When `flag` is 1, the constraint is automatically satisfied.
/// Degree: 2
#[inline]
pub fn assert_eq_when_off<F: IsField>(
    builder: &mut dyn AirBuilder<F>,
    flag: FieldElement<F>,
    left: FieldElement<F>,
    right: FieldElement<F>,
) {
    builder.assert_zero((FieldElement::<F>::one() - flag) * (left - right));
}

// =============================================================================
// Base-field helpers (MainAirBuilder<F, E>) - used by verifier path only
// =============================================================================

/// IS_BIT in base field: x*(1-x) == 0
#[inline]
pub fn assert_is_bit_base<F: IsSubFieldOf<E> + IsField, E: IsField>(
    builder: &mut dyn MainAirBuilder<F, E>,
    x: FieldElement<F>,
) {
    let one = FieldElement::<F>::one();
    builder.assert_zero_base(x.clone() * (one - x));
}

// =============================================================================
// Direct-evaluation helpers - used for prover main trace constraints
// =============================================================================
// These helpers compute constraint values directly without any trait dispatch.
// They return the constraint value to be stored in the evals buffer.

/// IS_BIT: x*(1-x). Returns the constraint value.
#[inline]
pub fn is_bit<F: IsField>(x: &FieldElement<F>) -> FieldElement<F> {
    x * &(FieldElement::<F>::one() - x)
}
