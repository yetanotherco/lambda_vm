//! Tests for the LT (Less-Than) table.

use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::lt::{LtOperation, bus_interactions, cols, generate_lt_trace, lt_constraints};
use crate::tables::types::FE;
use crate::test_utils::{busless_air, create_lt_air, in_chip_constraint_count, validate_busless};

/// Signed comparison flag
const SIGNED: bool = true;
/// Unsigned comparison flag
const UNSIGNED: bool = false;

#[test]
fn test_lt_unsigned_basic() {
    let ops = [
        LtOperation::new(5, 10, UNSIGNED),       // 5 < 10 unsigned -> true
        LtOperation::new(10, 5, UNSIGNED),       // 10 < 5 unsigned -> false
        LtOperation::new(5, 5, UNSIGNED),        // 5 < 5 unsigned -> false
        LtOperation::new(0, 1, UNSIGNED),        // 0 < 1 unsigned -> true
        LtOperation::new(u64::MAX, 0, UNSIGNED), // MAX < 0 unsigned -> false
    ];

    assert!(ops[0].compute_lt());
    assert!(!ops[1].compute_lt());
    assert!(!ops[2].compute_lt());
    assert!(ops[3].compute_lt());
    assert!(!ops[4].compute_lt());
}

#[test]
fn test_lt_signed_basic() {
    let ops = [
        LtOperation::new(5, 10, SIGNED),             // 5 < 10 signed -> true
        LtOperation::new(10, 5, SIGNED),             // 10 < 5 signed -> false
        LtOperation::new((-5i64) as u64, 5, SIGNED), // -5 < 5 signed -> true
        LtOperation::new(5, (-5i64) as u64, SIGNED), // 5 < -5 signed -> false
        LtOperation::new((-10i64) as u64, (-5i64) as u64, SIGNED), // -10 < -5 signed -> true
    ];

    assert!(ops[0].compute_lt());
    assert!(!ops[1].compute_lt());
    assert!(ops[2].compute_lt());
    assert!(!ops[3].compute_lt());
    assert!(ops[4].compute_lt());
}

#[test]
fn test_trace_generation() {
    let ops = vec![
        LtOperation::new(100, 200, UNSIGNED),
        LtOperation::new(200, 100, SIGNED),
    ];

    let trace = generate_lt_trace(&ops);

    // Should be padded to power of 2 (minimum 4 for FRI)
    assert_eq!(trace.main_table.height, 4);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    // Find the row with lhs=100 (HashMap ordering is not deterministic)
    let mut found_100_200 = false;
    let mut found_200_100 = false;

    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::LHS_0] == FE::from(100u64) && row[cols::RHS_0] == FE::from(200u64) {
            assert_eq!(row[cols::SIGNED], FE::zero());
            assert_eq!(row[cols::LT], FE::one()); // 100 < 200
            assert_eq!(row[cols::MU], FE::one()); // multiplicity = 1
            found_100_200 = true;
        }
        if row[cols::LHS_0] == FE::from(200u64) && row[cols::RHS_0] == FE::from(100u64) {
            assert_eq!(row[cols::SIGNED], FE::one());
            assert_eq!(row[cols::LT], FE::zero()); // 200 < 100 signed -> false
            assert_eq!(row[cols::MU], FE::one()); // multiplicity = 1
            found_200_100 = true;
        }
    }

    assert!(found_100_200, "Row with lhs=100, rhs=200 not found");
    assert!(found_200_100, "Row with lhs=200, rhs=100 not found");
}

#[test]
fn test_multiplicity_aggregation() {
    // Create 5 operations where (5, 10, UNSIGNED) appears 3 times
    let ops = vec![
        LtOperation::new(5, 10, UNSIGNED), // appears 1st time
        LtOperation::new(100, 200, UNSIGNED),
        LtOperation::new(5, 10, UNSIGNED),    // appears 2nd time
        LtOperation::new(5, 10, UNSIGNED),    // appears 3rd time
        LtOperation::new(100, 200, UNSIGNED), // duplicate
    ];

    let trace = generate_lt_trace(&ops);

    // Should deduplicate to 2 unique rows, padded to 4 (minimum for FRI)
    assert_eq!(trace.main_table.height, 4);

    // Find each unique operation and check multiplicity
    let mut found_5_10 = false;
    let mut found_100_200 = false;

    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::LHS_0] == FE::from(5u64) && row[cols::RHS_0] == FE::from(10u64) {
            assert_eq!(
                row[cols::MU],
                FE::from(3u64),
                "Expected multiplicity 3 for (5, 10)"
            );
            found_5_10 = true;
        }
        if row[cols::LHS_0] == FE::from(100u64) && row[cols::RHS_0] == FE::from(200u64) {
            assert_eq!(
                row[cols::MU],
                FE::from(2u64),
                "Expected multiplicity 2 for (100, 200)"
            );
            found_100_200 = true;
        }
    }

    assert!(found_5_10, "Row with lhs=5, rhs=10 not found");
    assert!(found_100_200, "Row with lhs=100, rhs=200 not found");
}

#[test]
fn test_multiplicity_different_signed_flags() {
    // Same lhs/rhs but different signed flag should be separate rows
    let ops = vec![
        LtOperation::new(5, 10, UNSIGNED), // unsigned
        LtOperation::new(5, 10, SIGNED),   // signed - different operation!
        LtOperation::new(5, 10, UNSIGNED), // unsigned again
    ];

    let trace = generate_lt_trace(&ops);

    // Should have 2 unique rows (unsigned and signed), padded to 4 (minimum for FRI)
    assert_eq!(trace.main_table.height, 4);

    let mut unsigned_mu = None;
    let mut signed_mu = None;

    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::LHS_0] == FE::from(5u64) && row[cols::RHS_0] == FE::from(10u64) {
            if row[cols::SIGNED] == FE::zero() {
                unsigned_mu = Some(row[cols::MU]);
            } else {
                signed_mu = Some(row[cols::MU]);
            }
        }
    }

    assert_eq!(
        unsigned_mu,
        Some(FE::from(2u64)),
        "Unsigned (5,10) should have mu=2"
    );
    assert_eq!(signed_mu, Some(FE::one()), "Signed (5,10) should have mu=1");
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();
    // MSB16 x2 + IS_HALFWORD x6 (lhs_sub_rhs x4 + lhs[1] + rhs[1]) + LT x1 = 9 interactions
    assert_eq!(interactions.len(), 9);
}

// Soundness regression: `lt` must equal `(lhs < rhs)`. The in-chip constraints were
// dead code until they were wired into the production `create_lt_air`, so a prover
// could certify a false comparison (and, via the memory-timestamp LT bus, forge
// memory consistency). These guard against reintroducing that hole.

/// Enforcement: a forged `lt = 1` for `20 <u 10` (true result 0) is rejected by
/// `LtFormula`, evaluated in isolation over a bus-less AIR.
#[test]
fn test_lt_rejects_false_comparison() {
    let air = busless_air(cols::NUM_COLUMNS, lt_constraints(0).0);
    let mut trace = generate_lt_trace(&[LtOperation::new(20, 10, UNSIGNED)]);
    assert!(
        validate_busless(&air, &trace),
        "honest LT row (20 <u 10 = 0) must validate"
    );

    trace.set_main(0, cols::LT, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "forged lt=1 for 20<u10 must be rejected by LtFormula"
    );
}

/// Wiring: `create_lt_air` registers the in-chip constraints on top of its bus
/// constraints. Directly catches a revert to `transition_constraints = vec![]`.
#[test]
fn test_lt_air_wires_in_chip_constraints() {
    let air = create_lt_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        cols::NUM_COLUMNS,
        bus_interactions(),
    );
    assert_eq!(in_chip, lt_constraints(0).0.len());
    assert_eq!(lt_constraints(0).0.len(), 3);
}
