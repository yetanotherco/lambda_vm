//! Tests for the MUL (Multiplication) table.

use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::mul::{
    MulOperation, bus_interactions, cols, generate_mul_trace, mul_constraints,
};
use crate::tables::types::FE;
use crate::test_utils::{
    busless_air, create_mul_air, in_chip_constraint_count, is_halfword_sender_columns,
    validate_busless,
};

#[test]
fn test_mul_unsigned_basic() {
    // Simple unsigned multiplications
    let op1 = MulOperation::new(5, false, 10, false);
    let (lo, hi) = op1.compute_product();
    assert_eq!(lo, 50);
    assert_eq!(hi, 0);

    // Larger numbers
    let op2 = MulOperation::new(0x1_0000_0000, false, 0x1_0000_0000, false);
    let (lo, hi) = op2.compute_product();
    assert_eq!(lo, 0); // 2^64 mod 2^64 = 0
    assert_eq!(hi, 1); // 2^64 / 2^64 = 1

    // Max values
    let op3 = MulOperation::new(u64::MAX, false, 2, false);
    let (lo, hi) = op3.compute_product();
    // u64::MAX * 2 = 2^65 - 2 = 0xFFFF_FFFF_FFFF_FFFE (lo) with hi = 1
    assert_eq!(lo, u64::MAX - 1);
    assert_eq!(hi, 1);
}

#[test]
fn test_mul_signed_basic() {
    // Positive * positive
    let op1 = MulOperation::new(5, true, 10, true);
    let (lo, hi) = op1.compute_product();
    assert_eq!(lo, 50);
    assert_eq!(hi, 0);

    // Negative * positive
    let op2 = MulOperation::new((-5i64) as u64, true, 10, true);
    let (lo, hi) = op2.compute_product();
    assert_eq!(lo as i64, -50);
    assert_eq!(hi, u64::MAX); // Sign extension

    // Negative * negative
    let op3 = MulOperation::new((-5i64) as u64, true, (-10i64) as u64, true);
    let (lo, hi) = op3.compute_product();
    assert_eq!(lo, 50);
    assert_eq!(hi, 0);

    // Large negative
    let op4 = MulOperation::new((-1i64) as u64, true, (-1i64) as u64, true);
    let (lo, hi) = op4.compute_product();
    assert_eq!(lo, 1);
    assert_eq!(hi, 0);
}

#[test]
fn test_mul_mulhsu() {
    // MULHSU: signed lhs * unsigned rhs
    // -1 (signed) * 2 (unsigned) = -2
    let op1 = MulOperation::new((-1i64) as u64, true, 2, false);
    let (lo, hi) = op1.compute_product();
    assert_eq!(lo as i64, -2);
    assert_eq!(hi, u64::MAX); // Sign extension

    // Positive signed * unsigned
    let op2 = MulOperation::new(5, true, 10, false);
    let (lo, hi) = op2.compute_product();
    assert_eq!(lo, 50);
    assert_eq!(hi, 0);
}

#[test]
fn test_sign_detection() {
    // Unsigned: never negative
    let op1 = MulOperation::new((-1i64) as u64, false, (-1i64) as u64, false);
    assert!(!op1.lhs_is_negative());
    assert!(!op1.rhs_is_negative());

    // Signed: check sign bit
    let op2 = MulOperation::new((-1i64) as u64, true, 5, true);
    assert!(op2.lhs_is_negative());
    assert!(!op2.rhs_is_negative());

    // MULHSU: lhs signed, rhs unsigned
    let op3 = MulOperation::new((-1i64) as u64, true, (-1i64) as u64, false);
    assert!(op3.lhs_is_negative());
    assert!(!op3.rhs_is_negative()); // rhs_signed=false, so never negative
}

#[test]
fn test_sign_extension() {
    // Positive number: no sign extension
    let op1 = MulOperation::new(0x1234_5678_9ABC_DEF0, false, 0, false);
    let ext = op1.lhs_extended();
    assert_eq!(ext[0], 0xDEF0);
    assert_eq!(ext[1], 0x9ABC);
    assert_eq!(ext[2], 0x5678);
    assert_eq!(ext[3], 0x1234);
    assert_eq!(ext[4], 0); // No extension
    assert_eq!(ext[5], 0);
    assert_eq!(ext[6], 0);
    assert_eq!(ext[7], 0);

    // Signed positive: still no extension
    let op2 = MulOperation::new(0x1234_5678_9ABC_DEF0, true, 0, false);
    let ext = op2.lhs_extended();
    assert_eq!(ext[4], 0); // MSB is 0, so positive
    assert_eq!(ext[7], 0);

    // Signed negative: full extension
    let op3 = MulOperation::new((-1i64) as u64, true, 0, false);
    let ext = op3.lhs_extended();
    assert_eq!(ext[0], 0xFFFF);
    assert_eq!(ext[3], 0xFFFF);
    assert_eq!(ext[4], 0xFFFF); // Sign extension
    assert_eq!(ext[7], 0xFFFF);
}

#[test]
fn test_raw_products() {
    // Simple case: 2 * 3 = 6
    let op = MulOperation::new(2, false, 3, false);
    let raw = op.compute_raw_products();
    // raw[0] should contain the low 32-bit portion of convolution
    // For 2 * 3, result is 6, which fits in raw[0]
    assert!(raw[0] >= 6, "raw[0] should contain product term");
}

#[test]
fn test_trace_generation() {
    let ops = vec![
        (MulOperation::new(100, false, 200, false), false), // wants_lo
        (MulOperation::new(5, true, 10, true), true),       // wants_hi
    ];

    let trace = generate_mul_trace(&ops);

    // Should be padded to power of 2 (minimum 4 for FRI)
    assert_eq!(trace.main_table.height, 4);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    // Find the rows
    let mut found_100_200 = false;
    let mut found_5_10 = false;

    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::LHS_0] == FE::from(100u64) && row[cols::RHS_0] == FE::from(200u64) {
            assert_eq!(row[cols::LHS_SIGNED], FE::zero());
            assert_eq!(row[cols::RHS_SIGNED], FE::zero());
            assert_eq!(row[cols::MU_LO], FE::one()); // wants_lo
            assert_eq!(row[cols::MU_HI], FE::zero());
            // Check product: 100 * 200 = 20000
            assert_eq!(row[cols::LO_0], FE::from(20000u64 & 0xFFFF));
            found_100_200 = true;
        }
        if row[cols::LHS_0] == FE::from(5u64) && row[cols::RHS_0] == FE::from(10u64) {
            assert_eq!(row[cols::LHS_SIGNED], FE::one());
            assert_eq!(row[cols::RHS_SIGNED], FE::one());
            assert_eq!(row[cols::MU_LO], FE::zero());
            assert_eq!(row[cols::MU_HI], FE::one()); // wants_hi
            // Check product: 5 * 10 = 50
            assert_eq!(row[cols::LO_0], FE::from(50u64));
            found_5_10 = true;
        }
    }

    assert!(found_100_200, "Row with lhs=100, rhs=200 not found");
    assert!(found_5_10, "Row with lhs=5, rhs=10 not found");
}

#[test]
fn test_multiplicity_aggregation() {
    // Create operations where same (lhs, rhs, signs) appears multiple times
    // with different wants_hi flags
    let ops = vec![
        (MulOperation::new(5, false, 10, false), false), // wants_lo
        (MulOperation::new(5, false, 10, false), false), // wants_lo again
        (MulOperation::new(5, false, 10, false), true),  // wants_hi
        (MulOperation::new(100, false, 200, false), true), // different op
    ];

    let trace = generate_mul_trace(&ops);

    // Should deduplicate to 2 unique rows, padded to 4
    assert_eq!(trace.main_table.height, 4);

    let mut found_5_10 = false;
    let mut found_100_200 = false;

    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::LHS_0] == FE::from(5u64) && row[cols::RHS_0] == FE::from(10u64) {
            assert_eq!(
                row[cols::MU_LO],
                FE::from(2u64),
                "Expected mu_lo=2 for (5, 10)"
            );
            assert_eq!(
                row[cols::MU_HI],
                FE::from(1u64),
                "Expected mu_hi=1 for (5, 10)"
            );
            found_5_10 = true;
        }
        if row[cols::LHS_0] == FE::from(100u64) && row[cols::RHS_0] == FE::from(200u64) {
            assert_eq!(
                row[cols::MU_LO],
                FE::zero(),
                "Expected mu_lo=0 for (100, 200)"
            );
            assert_eq!(
                row[cols::MU_HI],
                FE::one(),
                "Expected mu_hi=1 for (100, 200)"
            );
            found_100_200 = true;
        }
    }

    assert!(found_5_10, "Row with lhs=5, rhs=10 not found");
    assert!(found_100_200, "Row with lhs=100, rhs=200 not found");
}

#[test]
fn test_different_signed_flags_separate_rows() {
    // Same lhs/rhs but different signed flags should be separate rows
    let ops = vec![
        (MulOperation::new(5, false, 10, false), false), // UNSIGNED
        (MulOperation::new(5, true, 10, true), false),   // SIGNED
        (MulOperation::new(5, true, 10, false), false),  // MULHSU
    ];

    let trace = generate_mul_trace(&ops);

    // 3 unique operations, padded to 4
    assert_eq!(trace.main_table.height, 4);

    let mut count_5_10 = 0;
    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::LHS_0] == FE::from(5u64) && row[cols::RHS_0] == FE::from(10u64) {
            // Check that multiplicity is 1 for each unique operation
            assert_eq!(row[cols::MU_LO], FE::one());
            count_5_10 += 1;
        }
    }

    assert_eq!(
        count_5_10, 3,
        "Should have 3 separate rows for (5, 10) with different signed flags"
    );
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();
    // Expected interactions (every MUL lookup goes through the unified ALU
    // bus — CPU MUL/MULH dispatch and dvrm's `d*q` consistency):
    // - 2x MSB16 senders (lhs sign, rhs sign)
    // - 8x IS_HALF senders for inputs (lhs[0..4], rhs[0..4]) — range-check input halves
    // - 8x IS_HALF senders for outputs (lo[0..4], hi[0..4])
    // - 4x IS_B20 senders (carry[0..4] virtual range checks)
    // - 2x ALU receivers (lo, hi)
    // Total: 2 + 8 + 8 + 4 + 2 = 24
    assert_eq!(interactions.len(), 24, "Expected 24 bus interactions");
}

#[test]
fn test_large_multiplication() {
    // Test with large numbers that overflow into hi
    let op = MulOperation::new(0xFFFF_FFFF_FFFF_FFFF, false, 0xFFFF_FFFF_FFFF_FFFF, false);
    let (lo, hi) = op.compute_product();

    // (2^64 - 1)^2 = 2^128 - 2^65 + 1
    // lo = 1, hi = 2^64 - 2
    assert_eq!(lo, 1);
    assert_eq!(hi, 0xFFFF_FFFF_FFFF_FFFE);
}

#[test]
fn test_zero_multiplication() {
    let op1 = MulOperation::new(0, false, 12345, false);
    let (lo, hi) = op1.compute_product();
    assert_eq!(lo, 0);
    assert_eq!(hi, 0);

    let op2 = MulOperation::new(12345, true, 0, true);
    let (lo, hi) = op2.compute_product();
    assert_eq!(lo, 0);
    assert_eq!(hi, 0);
}

#[test]
fn test_identity_multiplication() {
    let op = MulOperation::new(12345, false, 1, false);
    let (lo, hi) = op.compute_product();
    assert_eq!(lo, 12345);
    assert_eq!(hi, 0);
}

// Soundness regression: the output must equal the product of the inputs, and the
// input halves must be range-checked. The in-chip constraints were dead code until
// they were wired into `create_mul_air`, so a prover could certify a false product.

/// Enforcement: a forged raw product (`20 * 20` claimed as 999) is rejected by the
/// `RawProduct` convolution, evaluated in isolation over a bus-less AIR.
#[test]
fn test_mul_rejects_false_product() {
    let air = busless_air(cols::NUM_COLUMNS, mul_constraints(0).0);
    let mut trace = generate_mul_trace(&[(MulOperation::new(20, false, 20, false), false)]);
    assert!(
        validate_busless(&air, &trace),
        "honest MUL row (20 * 20 = 400) must validate"
    );

    trace.set_main(0, cols::RAW_PRODUCT_0, FE::from(999u64));
    assert!(
        !validate_busless(&air, &trace),
        "forged raw product must be rejected by RawProduct"
    );
}

/// Wiring: `create_mul_air` registers the in-chip constraints on top of its bus
/// constraints. Directly catches a revert to `transition_constraints = vec![]`.
#[test]
fn test_mul_air_wires_in_chip_constraints() {
    let air = create_mul_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        cols::NUM_COLUMNS,
        bus_interactions(),
    );
    assert_eq!(in_chip, mul_constraints(0).0.len());
    // 2x SignedIsBit + LhsSign + RhsSign + 4x RawProduct (#644 added the two
    // SignedIsBit constraints that #652's count of 6 predated).
    assert_eq!(mul_constraints(0).0.len(), 8);
}

/// Presence: every input halfword is range-checked via IS_HALFWORD senders, so a
/// field-wrapping decomposition that keeps the packed word constant cannot change
/// the product undetected.
#[test]
fn test_mul_range_checks_input_halves() {
    let cols_checked = is_halfword_sender_columns(&bus_interactions());
    for c in [
        cols::LHS_0,
        cols::LHS_1,
        cols::LHS_2,
        cols::LHS_3,
        cols::RHS_0,
        cols::RHS_1,
        cols::RHS_2,
        cols::RHS_3,
    ] {
        assert!(
            cols_checked.contains(&c),
            "MUL must IS_HALF range-check input half column {c}"
        );
    }
}

/// Regression test for the `Msb16` LogUp over-send bug.
///
/// MUL is split into chip instances of `max_rows.mul` raw ops (`chunk_and_generate`)
/// and each instance deduplicates only its own chunk, sending the MSB16 sign lookup
/// once per unique signed op *per instance* (multiplicity = the `SIGNED` bit). So
/// `collect_bitwise_from_mul`, which feeds the BITWISE MSB16 multiplicity, must use
/// the *same* per-chunk dedup: a unique signed op spanning two instances is sent
/// twice but, with a single global dedup, would be tallied once — leaving the `Msb16`
/// bus unbalanced and verification failing for any block large enough to split MUL.
#[test]
fn msb16_bitwise_multiplicity_matches_per_instance_sends() {
    use crate::tables::bitwise::BitwiseOperationType;
    use crate::tables::trace_builder::collect_bitwise_from_mul;

    let chunk = 4usize;
    // One unique signed mul op repeated so it spans two `chunk`-sized instances.
    let op = MulOperation::new(0x1234_5678_9abc_def0, true, 0x0fed_cba9_8765_4321, true);
    let ops: Vec<(MulOperation, bool)> = std::iter::repeat_n((op, false), 6).collect();
    assert!(
        ops.len() > chunk,
        "scenario must split MUL into >1 instance"
    );

    // Ground truth: each instance is one chunk, and the MUL AIR sends MSB16 with
    // multiplicity Column(LHS_SIGNED) (lhs) and Column(RHS_SIGNED) (rhs) per row, so
    // total sends = Σ rows (LHS_SIGNED + RHS_SIGNED).
    let mut sends = 0usize;
    for c in ops.chunks(chunk) {
        let trace = generate_mul_trace(c);
        for row in 0..trace.num_rows() {
            sends += (*trace.get_main(row, cols::LHS_SIGNED) == FE::one()) as usize;
            sends += (*trace.get_main(row, cols::RHS_SIGNED) == FE::one()) as usize;
        }
    }
    assert_eq!(sends, 4, "sanity: 2 instances × (lhs + rhs) sends");

    let tallied = collect_bitwise_from_mul(&ops, chunk)
        .iter()
        .filter(|b| matches!(b.lookup_type, BitwiseOperationType::Msb16))
        .count();
    assert_eq!(
        tallied, sends,
        "BITWISE MSB16 multiplicity ({tallied}) must equal total MUL-instance MSB16 \
         sends ({sends}); a mismatch leaves the Msb16 bus unbalanced"
    );
}
