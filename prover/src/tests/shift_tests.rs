//! Tests for the SHIFT table.

use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::shift::{NUM_SHIFT_CONSTRAINTS, bus_interactions, cols};
use crate::test_utils::{create_shift_air, in_chip_constraint_count, is_halfword_sender_columns};

// Soundness regression: every input halfword must be range-checked. That gap was
// closed by the in-chip soundness fix (VM-3). The `(1 - signed) * is_negative = 0`
// constraint (VM-4) is a spec-level gap (`shift.toml` omits it) and is deferred to a
// separate spec fix, so it is intentionally not enforced here.

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

/// Wiring: `create_shift_air` registers all in-chip constraints on top of its bus
/// constraints. Catches a revert to `transition_constraints = vec![]` or a dropped
/// constraint (the count would differ).
#[test]
fn test_shift_air_wires_in_chip_constraints() {
    let air = create_shift_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        cols::NUM_COLUMNS,
        bus_interactions(),
    );
    assert_eq!(in_chip, NUM_SHIFT_CONSTRAINTS);
    assert_eq!(NUM_SHIFT_CONSTRAINTS, 16);
}
