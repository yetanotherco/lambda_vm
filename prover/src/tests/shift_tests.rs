//! Tests for the SHIFT table.

use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::shift::{
    NUM_SHIFT_CONSTRAINTS, ShiftOperation, bus_interactions, cols, generate_shift_trace,
    shift_constraints,
};
use crate::tables::types::FE;
use crate::test_utils::{
    busless_air, create_shift_air, in_chip_constraint_count, is_halfword_sender_columns,
    validate_busless,
};

// Soundness regression: `is_negative` must be 0 on an unsigned (logical) shift, and
// every input halfword must be range-checked. Both gaps were closed by the in-chip
// soundness fix.

/// Enforcement: forging `is_negative = 1` on an unsigned `SRL` is rejected by
/// `IsNegativeZeroWhenUnsigned`. Before the fix `is_negative` was free for
/// `signed = 0`, letting a logical shift sign-extend.
#[test]
fn test_shift_rejects_is_negative_when_unsigned() {
    let air = busless_air(cols::NUM_COLUMNS, shift_constraints(0).0);
    // Unsigned (signed = false) right (direction = true) shift.
    let mut trace =
        generate_shift_trace(&[ShiftOperation::new(0x1234_5678, 8, true, false, false)]);
    assert!(
        validate_busless(&air, &trace),
        "honest unsigned SRL row must validate (is_negative = 0)"
    );

    trace.set_main(0, cols::IS_NEGATIVE, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "is_negative = 1 on an unsigned shift must be rejected"
    );
}

/// Presence: every input halfword is range-checked via IS_HALFWORD senders, so a
/// field-wrapping decomposition that keeps the packed operand constant cannot
/// change the shifted output undetected.
#[test]
fn test_shift_range_checks_input_halves() {
    let cols_checked = is_halfword_sender_columns(&bus_interactions());
    for c in cols::IN {
        assert!(
            cols_checked.contains(&c),
            "SHIFT must IS_HALF range-check input half column {c}"
        );
    }
}

/// Wiring: `create_shift_air` registers all in-chip constraints (including
/// `IsNegativeZeroWhenUnsigned`) on top of its bus constraints. Catches a revert to
/// `transition_constraints = vec![]` or a dropped constraint (the count would differ).
#[test]
fn test_shift_air_wires_in_chip_constraints() {
    let air = create_shift_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        cols::NUM_COLUMNS,
        bus_interactions(),
    );
    assert_eq!(in_chip, NUM_SHIFT_CONSTRAINTS);
    assert_eq!(NUM_SHIFT_CONSTRAINTS, 17);
}
