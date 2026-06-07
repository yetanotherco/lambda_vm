//! Soundness regression guards for the underconstrained-chip bugs (VM-1..VM-7).
//!
//! Background: an audit found that several chips were *underconstrained* — a
//! malicious prover could pick witness values not forced by the AIR. Commit
//! `7562b7e8 "register missing in-chip soundness constraints"` fixed the five
//! live / audit-flagged ones:
//!
//! | ID   | Chip  | What was missing                                    | Fix mechanism            |
//! |------|-------|-----------------------------------------------------|--------------------------|
//! | VM-1 | LT    | `lt_constraints` was dead code (`create_lt_air` had `vec![]`) | wire transition constraints |
//! | VM-2 | MUL   | `mul_constraints` was dead code; inputs not range-checked    | wire constraints + IS_HALF senders |
//! | VM-3 | SHIFT | input halves not range-checked when `zbs = 1`                | IS_HALF senders over `IN_0..IN_3` |
//! | VM-4 | SHIFT | `is_negative` free on an unsigned shift                       | `IsNegativeZeroWhenUnsigned` constraint |
//! | VM-5 | DVRM  | `remainder = numerator` not forced on division by zero       | `DivByZeroR` constraint + denom IS_HALF senders |
//!
//! These guards PASS on the fixed code and FAIL if a fix is reverted. They come
//! in three flavours, matching the two fix mechanisms:
//!
//! * **Enforcement** — a forged witness is rejected by the now-live transition
//!   constraint. We isolate the chip's transition constraints in a *bus-less*
//!   AIR (no LogUp aux columns) and run [`validate_trace`]. This works for the
//!   constraints that are pure transition constraints (LT `LtFormula`, MUL
//!   `RawProduct`, SHIFT `IsNegativeZeroWhenUnsigned`, DVRM `DivByZeroR`).
//! * **Wiring** — the production `create_*_air` builders register the in-chip
//!   constraints *on top of* their bus constraints. We assert the constraint
//!   count delta vs. a bus-only AIR equals the number of in-chip constraints.
//!   This is what directly catches the original `vec![]` regression (VM-1/VM-2),
//!   which a count-only check cannot, because `AirWithBuses::new` also appends
//!   LogUp constraints.
//! * **Presence** — the range-check fixes are LogUp *bus interactions*
//!   (`IS_HALFWORD` senders), invisible to `validate_trace`. We assert the
//!   senders exist over the relevant columns (VM-2 inputs, VM-3 SHIFT inputs,
//!   VM-5 denominator).
//!
//! VM-6 (SHIFT control-flag bit-ness) and VM-7 (LOAD flag bit-ness + width
//! exclusivity) are NOT fixed: they are *latent* — the CPU pins these flags to
//! bits via the preprocessed DECODE table, so they are not exploitable
//! end-to-end. They are recorded here as `#[ignore]`d, deferred defense-in-depth
//! (see `docs/soundness-bugs-shareable-repro.md`).

use math::field::element::FieldElement;

use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::debug::validate_trace;
use stark::domain::Domain;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, NullBoundaryConstraintBuilder,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;

use crate::tables::dvrm::{
    DvrmOperation, bus_interactions as dvrm_bus_interactions, cols as dvrm_cols, dvrm_constraints,
    generate_dvrm_trace,
};
use crate::tables::lt::{
    LtOperation, bus_interactions as lt_bus_interactions, cols as lt_cols, generate_lt_trace,
    lt_constraints,
};
use crate::tables::mul::{
    MulOperation, bus_interactions as mul_bus_interactions, cols as mul_cols, generate_mul_trace,
    mul_constraints,
};
use crate::tables::shift::{
    NUM_SHIFT_CONSTRAINTS, ShiftOperation, bus_interactions as shift_bus_interactions,
    cols as shift_cols, generate_shift_trace, shift_constraints,
};
use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};
use crate::test_utils::{create_lt_air, create_mul_air, create_shift_air};

type F = GoldilocksField;
type E = GoldilocksExtension;
type ChipAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()>;

// =============================================================================
// Helpers
// =============================================================================

/// Build a bus-less AIR carrying only the given in-chip transition constraints.
/// With zero bus interactions, `AirWithBuses::new` appends no LogUp constraints
/// and allocates no aux columns, so `validate_trace` evaluates exactly the
/// chip's transition constraints over a main-only trace.
fn busless_air(
    num_columns: usize,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
) -> ChipAir {
    AirWithBuses::new(
        num_columns,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        &ProofOptions::default_test_options(),
        1,
        constraints,
    )
}

/// Run `validate_trace` for a bus-less chip AIR over a main-only trace.
/// Returns `true` iff every transition constraint holds on every row.
fn validate_busless(air: &ChipAir, trace: &TraceTable<F, E>) -> bool {
    let domain = Domain::new(air, trace.num_rows());
    validate_trace(air, &(), trace, &domain, &[], None)
}

/// Number of transition constraints the production builder registers on top of
/// its bus constraints, computed as a delta against a bus-only AIR with the same
/// interactions but no in-chip constraints. This isolates the in-chip count even
/// though `AirWithBuses::new` also appends LogUp constraints.
fn in_chip_constraint_count(wired: usize, num_columns: usize, buses: Vec<BusInteraction>) -> usize {
    let bus_only = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        num_columns,
        AuxiliaryTraceBuildData {
            interactions: buses,
        },
        &ProofOptions::default_test_options(),
        1,
        vec![],
    )
    .num_transition_constraints();
    wired - bus_only
}

/// Collect the `start_column`s of every `IS_HALFWORD` *sender* in `interactions`.
fn is_halfword_sender_columns(interactions: &[BusInteraction]) -> Vec<usize> {
    let id: u64 = BusId::IsHalfword.into();
    interactions
        .iter()
        .filter(|i| i.is_sender && i.bus_id == id)
        .flat_map(|i| {
            i.values.iter().filter_map(|v| match v {
                BusValue::Packed { start_column, .. } => Some(*start_column),
                BusValue::Linear(_) => None,
            })
        })
        .collect()
}

fn boxed<C: TransitionConstraint<F, E> + 'static>(
    cs: Vec<C>,
) -> Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> {
    cs.into_iter().map(|c| c.boxed()).collect()
}

// =============================================================================
// VM-1 — LT: `lt` must equal `(lhs < rhs)`
// =============================================================================

/// Enforcement: a forged `lt = 1` for `20 <u 10` (true result 0) is rejected by
/// `LtFormula`. Before the fix this validated (LT had zero in-chip constraints).
#[test]
fn lt_formula_rejects_false_comparison() {
    let air = busless_air(lt_cols::NUM_COLUMNS, boxed(lt_constraints(0).0));
    let mut trace = generate_lt_trace(&[LtOperation::new(20, 10, false)]);

    assert!(
        validate_busless(&air, &trace),
        "honest LT row (20 <u 10 = 0) must validate"
    );

    // Forge the output: claim 20 < 10.
    trace.set_main(0, lt_cols::LT, FieldElement::one());
    assert!(
        !validate_busless(&air, &trace),
        "VM-1 regression: forged lt=1 for 20<u10 must be rejected by LtFormula"
    );
}

/// Wiring: the production `create_lt_air` registers the 3 in-chip constraints on
/// top of its bus constraints. Directly catches a revert to `vec![]`.
#[test]
fn lt_air_wires_in_chip_constraints() {
    let air = create_lt_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        lt_cols::NUM_COLUMNS,
        lt_bus_interactions(),
    );
    assert_eq!(
        in_chip,
        lt_constraints(0).0.len(),
        "VM-1 regression: create_lt_air must wire its in-chip constraints (was vec![])"
    );
    assert_eq!(
        lt_constraints(0).0.len(),
        3,
        "LT defines 3 in-chip constraints"
    );
}

// =============================================================================
// VM-2 — MUL: output must equal the product of the inputs; inputs range-checked
// =============================================================================

/// Enforcement: a forged raw product (`20 * 20` claimed as 999) is rejected by
/// the `RawProduct` convolution. Before the fix MUL had zero in-chip constraints.
#[test]
fn mul_convolution_rejects_false_product() {
    let air = busless_air(mul_cols::NUM_COLUMNS, boxed(mul_constraints(0).0));
    let mut trace = generate_mul_trace(&[(MulOperation::new(20, false, 20, false), false)]);

    assert!(
        validate_busless(&air, &trace),
        "honest MUL row (20 * 20 = 400) must validate"
    );

    // Forge the convolution output limb.
    trace.set_main(0, mul_cols::RAW_PRODUCT_0, FieldElement::from(999u64));
    assert!(
        !validate_busless(&air, &trace),
        "VM-2 regression: forged raw product must be rejected by RawProduct"
    );
}

/// Wiring: `create_mul_air` registers the 6 in-chip constraints.
#[test]
fn mul_air_wires_in_chip_constraints() {
    let air = create_mul_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        mul_cols::NUM_COLUMNS,
        mul_bus_interactions(),
    );
    assert_eq!(
        in_chip,
        mul_constraints(0).0.len(),
        "VM-2 regression: create_mul_air must wire its in-chip constraints (was vec![])"
    );
    assert_eq!(
        mul_constraints(0).0.len(),
        6,
        "MUL defines 6 in-chip constraints"
    );
}

/// Presence: MUL range-checks every input halfword via IS_HALFWORD senders, so a
/// field-wrapping half decomposition that keeps the packed word constant cannot
/// change the product undetected.
#[test]
fn mul_bus_range_checks_input_halves() {
    let cols = is_halfword_sender_columns(&mul_bus_interactions());
    for c in [
        mul_cols::LHS_0,
        mul_cols::LHS_1,
        mul_cols::LHS_2,
        mul_cols::LHS_3,
        mul_cols::RHS_0,
        mul_cols::RHS_1,
        mul_cols::RHS_2,
        mul_cols::RHS_3,
    ] {
        assert!(
            cols.contains(&c),
            "VM-2 regression: MUL must IS_HALF range-check input half column {c}"
        );
    }
}

// =============================================================================
// VM-3 — SHIFT: every input halfword must be range-checked
// =============================================================================

/// Presence: SHIFT range-checks `IN_0..IN_3` via IS_HALFWORD senders. The fix is
/// a bus interaction (not a transition constraint), so it is checked structurally.
#[test]
fn shift_bus_range_checks_all_input_halves() {
    let cols = is_halfword_sender_columns(&shift_bus_interactions());
    for c in shift_cols::IN {
        assert!(
            cols.contains(&c),
            "VM-3 regression: SHIFT must IS_HALF range-check input half column {c}"
        );
    }
}

// =============================================================================
// VM-4 — SHIFT: `is_negative` must be 0 on an unsigned (logical) shift
// =============================================================================

/// Enforcement: forging `is_negative = 1` on an unsigned `SRL` is rejected by
/// `IsNegativeZeroWhenUnsigned`. Before the fix `is_negative` was free for
/// `signed = 0`, allowing a logical shift to sign-extend.
#[test]
fn shift_is_negative_rejected_when_unsigned() {
    let air = busless_air(shift_cols::NUM_COLUMNS, boxed(shift_constraints(0).0));
    // Unsigned (`signed = false`) right (`direction = true`) shift.
    let mut trace =
        generate_shift_trace(&[ShiftOperation::new(0x1234_5678, 8, true, false, false)]);

    assert!(
        validate_busless(&air, &trace),
        "honest unsigned SRL row must validate (is_negative = 0)"
    );

    trace.set_main(0, shift_cols::IS_NEGATIVE, FieldElement::one());
    assert!(
        !validate_busless(&air, &trace),
        "VM-4 regression: is_negative=1 on an unsigned shift must be rejected"
    );
}

/// Wiring: `create_shift_air` registers all 17 in-chip constraints, including
/// `IsNegativeZeroWhenUnsigned`.
#[test]
fn shift_air_wires_in_chip_constraints() {
    let air = create_shift_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        shift_cols::NUM_COLUMNS,
        shift_bus_interactions(),
    );
    assert_eq!(
        in_chip, NUM_SHIFT_CONSTRAINTS,
        "VM-4 regression: create_shift_air must wire all SHIFT in-chip constraints"
    );
    assert_eq!(
        NUM_SHIFT_CONSTRAINTS, 17,
        "SHIFT defines 17 in-chip constraints"
    );
}

// =============================================================================
// VM-5 — DVRM: `remainder = numerator` on division by zero
// =============================================================================

/// Enforcement: on a division-by-zero row, forging `r != n` is rejected by
/// `DivByZeroR`. RISC-V `REM/REMU` by zero must return the numerator.
#[test]
fn dvrm_div_by_zero_remainder_pinned() {
    let air = busless_air(dvrm_cols::NUM_COLUMNS, boxed(dvrm_constraints(0).0));
    // numerator = 20, denominator = 0 => div-by-zero, honest remainder = 20.
    let mut trace = generate_dvrm_trace(&[(DvrmOperation::new(20, 0, false), true)]);

    assert!(
        validate_busless(&air, &trace),
        "honest div-by-zero row (r = n = 20) must validate"
    );

    trace.set_main(0, dvrm_cols::R_0, FieldElement::from(999u64));
    assert!(
        !validate_busless(&air, &trace),
        "VM-5 regression: a forged remainder on div-by-zero must be rejected"
    );
}

/// Presence: DVRM range-checks the denominator halves via IS_HALFWORD senders.
#[test]
fn dvrm_bus_range_checks_denominator_halves() {
    let cols = is_halfword_sender_columns(&dvrm_bus_interactions());
    for c in [
        dvrm_cols::D_0,
        dvrm_cols::D_1,
        dvrm_cols::D_2,
        dvrm_cols::D_3,
    ] {
        assert!(
            cols.contains(&c),
            "VM-5 regression: DVRM must IS_HALF range-check denominator half column {c}"
        );
    }
}

// =============================================================================
// VM-6 / VM-7 — deferred (latent, defense-in-depth). NOT fixed by 7562b7e8.
// =============================================================================

/// VM-6 (deferred): SHIFT does not constrain `direction`, `signed`, `word_instr`
/// to be bits at the chip level. Latent — the CPU pins them to bits via the
/// preprocessed DECODE table, so it is not exploitable end-to-end today.
/// Un-ignore and assert the bit-ness constraints once VM-6 is fixed.
#[test]
#[ignore = "VM-6 deferred: SHIFT control-flag bit-ness (latent, pinned by CPU->DECODE)"]
fn shift_control_flags_should_be_bit_constrained() {
    unimplemented!("VM-6 not fixed yet — defense-in-depth; see soundness-bugs-shareable-repro.md");
}

/// VM-7 (deferred): LOAD does not constrain `signed`/`read2`/`read4`/`read8` to
/// bits nor `read2 + read4 + read8` to a bit (mutual exclusivity). Latent — the
/// CPU pins these via the preprocessed DECODE value. Un-ignore once VM-7 is fixed.
#[test]
#[ignore = "VM-7 deferred: LOAD flag bit-ness + width exclusivity (latent, pinned by CPU->DECODE)"]
fn load_flags_should_be_bit_and_width_exclusive() {
    unimplemented!("VM-7 not fixed yet — defense-in-depth; see soundness-bugs-shareable-repro.md");
}
