//! Tests for the LT (Less-Than) table.

use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::lt::{LtConstraints, LtOperation, bus_interactions, cols, generate_lt_trace};
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
    // MSB16 x2 + IS_HALFWORD x6 (lhs_sub_rhs x4 + lhs[1] + rhs[1])
    // + ALU receiver x1 (every LT lookup goes through the unified ALU bus
    // — CPU SLT/BLT/BGE dispatch and the internal memw/dvrm
    // timestamp / |r|<|d| checks) + IS_HALFWORD x1 bounding μ = 10.
    assert_eq!(interactions.len(), 10);
}

// Soundness regression: `lt` must equal `(lhs < rhs)`. The in-chip constraints were
// dead code until they were wired into the production `create_lt_air`, so a prover
// could certify a false comparison (and, via the memory-timestamp LT bus, forge
// memory consistency). These guard against reintroducing that hole.

/// Enforcement: a forged `lt = 1` for `20 <u 10` (true result 0) is rejected by
/// `LtFormula`, evaluated in isolation over a bus-less AIR.
#[test]
fn test_lt_rejects_false_comparison() {
    let air = busless_air(cols::NUM_COLUMNS, LtConstraints);
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
    use stark::constraints::builder::ConstraintSet;
    assert_eq!(in_chip, LtConstraints.meta().len());
    // Carry0IsBit, Carry1IsBit, LtFormula, OutXorInvert, InvertIsBit, SignedIsBit.
    assert_eq!(LtConstraints.meta().len(), 6);
}

/// Enforcement (this branch's unified-ALU-bus layout): the bus consumes `out`,
/// not `lt`. A forged `out` (e.g. `out = 1` while `lt = invert = 0`) must be
/// rejected by `OutXorInvert`. This is the hole `LtFormula` alone does NOT close
/// here, since `LtFormula` only binds `lt`.
#[test]
fn test_lt_rejects_forged_out() {
    let air = busless_air(cols::NUM_COLUMNS, LtConstraints);
    // 20 <u 10 = 0 and invert = 0 ⇒ out must be 0.
    let mut trace = generate_lt_trace(&[LtOperation::new(20, 10, UNSIGNED)]);
    assert!(
        validate_busless(&air, &trace),
        "honest LT row (out = lt XOR invert = 0) must validate"
    );

    trace.set_main(0, cols::OUT, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "forged out=1 (lt=invert=0) must be rejected by OutXorInvert"
    );
}

// Soundness regression: μ is a bounded, non-negative multiplicity. Every LT
// lookup fires with μ, so a free μ = −1 twin of an honest row received range
// lookups instead of sending them and absorbed another row's out-of-range limb
// (`lt_multiplicity_poc`, `multiplicity_forgery_poc`). μ is now IS_HALF-checked
// weighted by itself, and trace generation splits a row whose count would
// exceed `MU_MAX`.

/// Presence: μ is IS_HALF-checked weighted by itself, so an out-of-range value
/// `v` always lands on the bus with weight `v ≠ 0`.
#[test]
fn test_lt_bounds_its_multiplicity() {
    use crate::tables::types::BusId;
    use stark::lookup::{BusValue, Multiplicity, Packing};
    assert!(
        bus_interactions().iter().any(|i| i.is_sender
            && i.bus_id == BusId::IsHalfword as u64
            && matches!(i.multiplicity, Multiplicity::Column(m) if m == cols::MU)
            && matches!(i.values.as_slice(),
                [BusValue::Packed { start_column, packing: Packing::Direct }] if *start_column == cols::MU)),
        "LT must IS_HALF-check μ weighted by itself"
    );
}

/// Splitting: a count above `MU_MAX` spreads over rows within the bound that
/// preserve the total; exactly `MU_MAX` still fits one row.
#[test]
fn test_dedup_lt_rows_splits_counts_above_mu_max() {
    use crate::tables::lt::{MU_MAX, dedup_lt_rows};
    let op = LtOperation::new(5, 3, SIGNED);
    let ops: Vec<_> = std::iter::repeat_n(op.clone(), 2 * MU_MAX as usize + 7).collect();
    let mut rows: Vec<u64> = dedup_lt_rows(&ops).iter().map(|(_, mu)| *mu).collect();
    rows.sort();
    assert_eq!(rows, vec![7, MU_MAX, MU_MAX]);

    let ops: Vec<_> = std::iter::repeat_n(op, MU_MAX as usize).collect();
    assert_eq!(dedup_lt_rows(&ops).len(), 1);
    assert!(dedup_lt_rows(&[]).is_empty());

    // The generated trace never holds a μ above the bound.
    let ops: Vec<_> =
        std::iter::repeat_n(LtOperation::new(1, 2, UNSIGNED), MU_MAX as usize + 1).collect();
    let trace = generate_lt_trace(&ops);
    let mut total = 0u64;
    for r in 0..trace.num_rows() {
        let mu = trace.get_main(r, cols::MU).to_raw();
        assert!(mu <= MU_MAX);
        total += mu;
    }
    assert_eq!(total, ops.len() as u64);
}

/// Consistency: the BITWISE collector tallies exactly the IS_HALF[μ] lookups the
/// generated LT instances send (each row's μ, weighted by μ; padding nothing),
/// across several small instances and for a row split by `MU_MAX`.
#[test]
fn test_lt_mu_bound_lookups_match_collector() {
    use crate::tables::bitwise::BitwiseOperationType;
    use crate::tables::lt::MU_MAX;
    use crate::tables::trace_builder::collect_bitwise_from_lt;
    use std::collections::HashMap;

    let half = |h: u64| ((h & 0xFF) as u8, ((h >> 8) & 0xFF) as u8);
    let distinct = [
        LtOperation::new(5, 3, SIGNED),
        LtOperation::new(u64::MAX, 1, UNSIGNED),
        LtOperation::new(0x1234_5678_9abc_def0, 0x1234_5678_0000_0000, SIGNED),
    ];
    let small: Vec<_> = distinct.iter().cycle().take(11).cloned().collect();
    let split: Vec<_> = std::iter::repeat_n(distinct[0].clone(), MU_MAX as usize + 2)
        .chain(distinct.iter().cloned())
        .collect();

    for (ops, chunk) in [(small, 4usize), (split, 1 << 20)] {
        let collected = collect_bitwise_from_lt(&ops, chunk);

        // IS_HALF minus the per-raw-op range checks leaves the μ bound.
        let mut is_half: HashMap<(u8, u8), i64> = HashMap::new();
        for b in collected
            .iter()
            .filter(|b| b.lookup_type == BitwiseOperationType::IsHalf)
        {
            *is_half.entry((b.x, b.y)).or_default() += 1;
        }
        for op in &ops {
            let sub = op.lhs.wrapping_sub(op.rhs);
            for shift in [0, 16, 32, 48] {
                *is_half.entry(half(sub >> shift & 0xFFFF)).or_default() -= 1;
            }
            *is_half.entry(half(op.lhs >> 32 & 0xFFFF)).or_default() -= 1;
            *is_half.entry(half(op.rhs >> 32 & 0xFFFF)).or_default() -= 1;
        }

        let mut expected: HashMap<(u8, u8), i64> = HashMap::new();
        for c in ops.chunks(chunk) {
            let t = generate_lt_trace(c);
            for r in 0..t.num_rows() {
                let mu = t.get_main(r, cols::MU).to_raw();
                *expected.entry(half(mu)).or_default() += mu as i64;
            }
        }
        is_half.retain(|_, n| *n != 0);
        expected.retain(|_, n| *n != 0);
        assert_eq!(
            is_half, expected,
            "IS_HALF[μ] tally must match the trace rows"
        );
    }
}
