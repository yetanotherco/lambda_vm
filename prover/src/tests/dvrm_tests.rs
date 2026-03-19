//! Tests for the DVRM (Division/Remainder) table.

use crate::tables::dvrm::{DvrmOperation, bus_interactions, cols, generate_dvrm_trace};
use crate::tables::types::FE;

/// Signed comparison flag
const SIGNED: bool = true;
/// Unsigned comparison flag
const UNSIGNED: bool = false;

#[test]
fn test_dvrm_unsigned_basic() {
    // Simple unsigned divisions
    let op1 = DvrmOperation::new(20, 3, UNSIGNED);
    assert_eq!(op1.compute_quotient(), 6);
    assert_eq!(op1.compute_remainder(), 2);

    // Exact division
    let op2 = DvrmOperation::new(100, 10, UNSIGNED);
    assert_eq!(op2.compute_quotient(), 10);
    assert_eq!(op2.compute_remainder(), 0);

    // Numerator < denominator
    let op3 = DvrmOperation::new(3, 10, UNSIGNED);
    assert_eq!(op3.compute_quotient(), 0);
    assert_eq!(op3.compute_remainder(), 3);

    // Large numbers
    let op4 = DvrmOperation::new(u64::MAX, 2, UNSIGNED);
    assert_eq!(op4.compute_quotient(), u64::MAX / 2);
    assert_eq!(op4.compute_remainder(), 1);
}

#[test]
fn test_dvrm_signed_basic() {
    // Positive / positive
    let op1 = DvrmOperation::new(20, 3, SIGNED);
    assert_eq!(op1.compute_quotient() as i64, 6);
    assert_eq!(op1.compute_remainder() as i64, 2);

    // Negative / positive
    let op2 = DvrmOperation::new((-20i64) as u64, 3, SIGNED);
    assert_eq!(op2.compute_quotient() as i64, -6);
    assert_eq!(op2.compute_remainder() as i64, -2);

    // Positive / negative
    let op3 = DvrmOperation::new(20, (-3i64) as u64, SIGNED);
    assert_eq!(op3.compute_quotient() as i64, -6);
    assert_eq!(op3.compute_remainder() as i64, 2);

    // Negative / negative
    let op4 = DvrmOperation::new((-20i64) as u64, (-3i64) as u64, SIGNED);
    assert_eq!(op4.compute_quotient() as i64, 6);
    assert_eq!(op4.compute_remainder() as i64, -2);
}

#[test]
fn test_div_by_zero() {
    // Unsigned div by zero: q = all 1s, r = n
    let op1 = DvrmOperation::new(42, 0, UNSIGNED);
    assert!(op1.is_div_by_zero());
    assert!(!op1.is_overflow());
    assert_eq!(op1.compute_quotient(), u64::MAX);
    assert_eq!(op1.compute_remainder(), 42);

    // Signed div by zero: same behavior
    let op2 = DvrmOperation::new((-5i64) as u64, 0, SIGNED);
    assert!(op2.is_div_by_zero());
    assert_eq!(op2.compute_quotient(), u64::MAX);
    assert_eq!(op2.compute_remainder(), (-5i64) as u64);

    // Zero / zero
    let op3 = DvrmOperation::new(0, 0, UNSIGNED);
    assert!(op3.is_div_by_zero());
    assert_eq!(op3.compute_quotient(), u64::MAX);
    assert_eq!(op3.compute_remainder(), 0);
}

#[test]
fn test_overflow() {
    // Signed overflow: MIN / -1
    let op = DvrmOperation::new(i64::MIN as u64, u64::MAX, SIGNED);
    assert!(op.is_overflow());
    assert!(!op.is_div_by_zero());
    assert_eq!(op.compute_quotient(), i64::MIN as u64);
    assert_eq!(op.compute_remainder(), 0);

    // Same values but unsigned: not overflow
    let op2 = DvrmOperation::new(i64::MIN as u64, u64::MAX, UNSIGNED);
    assert!(!op2.is_overflow());
    assert_eq!(op2.compute_quotient(), 0);
    assert_eq!(op2.compute_remainder(), i64::MIN as u64);
}

#[test]
fn test_sign_detection() {
    // Unsigned: never negative
    let op1 = DvrmOperation::new((-1i64) as u64, (-1i64) as u64, UNSIGNED);
    assert!(!op1.sign_n());
    assert!(!op1.sign_d());
    assert!(!op1.sign_r());
    assert!(!op1.sign_q());

    // Signed: positive numerator, positive denominator
    let op2 = DvrmOperation::new(20, 3, SIGNED);
    assert!(!op2.sign_n());
    assert!(!op2.sign_d());
    assert!(!op2.sign_r()); // remainder = 2 (positive)
    assert!(op2.sign_q()); // sign_q = signed && !overflow = true

    // Signed: negative numerator
    let op3 = DvrmOperation::new((-20i64) as u64, 3, SIGNED);
    assert!(op3.sign_n());
    assert!(!op3.sign_d());
    assert!(op3.sign_r()); // remainder = -2 (negative)
    assert!(op3.sign_q()); // sign_q = signed && !overflow

    // Overflow: sign_q = false
    let op4 = DvrmOperation::new(i64::MIN as u64, u64::MAX, SIGNED);
    assert!(op4.sign_n());
    assert!(op4.sign_d());
    assert!(!op4.sign_q()); // sign_q = signed && !overflow = false
}

#[test]
fn test_abs_values() {
    // Unsigned: abs_r = r, abs_d = d
    let op1 = DvrmOperation::new(20, 3, UNSIGNED);
    assert_eq!(op1.abs_r(), 2);
    assert_eq!(op1.abs_d(), 3);

    // Signed: negative remainder → absolute value
    let op2 = DvrmOperation::new((-20i64) as u64, 3, SIGNED);
    assert_eq!(op2.abs_r(), 2); // |-2| = 2
    assert_eq!(op2.abs_d(), 3);

    // Signed: negative denominator → absolute value
    let op3 = DvrmOperation::new(20, (-3i64) as u64, SIGNED);
    assert_eq!(op3.abs_r(), 2);
    assert_eq!(op3.abs_d(), 3); // |-3| = 3
}

#[test]
fn test_n_sub_r() {
    // 20 / 3 = 6 remainder 2, n - r = 20 - 2 = 18 = 6 * 3
    let op1 = DvrmOperation::new(20, 3, UNSIGNED);
    assert_eq!(op1.n_sub_r(), 18);
    assert!(!op1.sign_n_sub_r());

    // Signed: (-20) / 3 = -6 remainder -2, n - r = -20 - (-2) = -18
    let op2 = DvrmOperation::new((-20i64) as u64, 3, SIGNED);
    assert_eq!(op2.n_sub_r() as i64, -18);
    assert!(op2.sign_n_sub_r());
}

#[test]
fn test_trace_generation() {
    let ops = vec![
        (DvrmOperation::new(100, 7, UNSIGNED), false), // wants_quotient (q=14, r=2)
        (DvrmOperation::new((-20i64) as u64, 3, SIGNED), true), // wants_remainder (q=-6, r=-2)
    ];

    let trace = generate_dvrm_trace(&ops);

    // Should be padded to power of 2 (minimum 4 for FRI)
    assert_eq!(trace.main_table.height, 4);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    let mut found_100_7 = false;
    let mut found_neg20_3 = false;

    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);

        // Check unsigned 100 / 7
        if row[cols::N_0] == FE::from(100u64) && row[cols::D_0] == FE::from(7u64) {
            assert_eq!(row[cols::SIGNED], FE::zero());
            assert_eq!(row[cols::MU_Q], FE::one()); // wants_quotient
            assert_eq!(row[cols::MU_R], FE::zero());
            // q = 14: halfwords (14, 0, 0, 0)
            assert_eq!(row[cols::Q_0], FE::from(14u64));
            assert_eq!(row[cols::Q_1], FE::zero());
            // r = 2: halfwords (2, 0, 0, 0)
            assert_eq!(row[cols::R_0], FE::from(2u64));
            assert_eq!(row[cols::R_1], FE::zero());
            assert_eq!(row[cols::DIV_BY_ZERO], FE::zero());
            assert_eq!(row[cols::OVERFLOW], FE::zero());
            found_100_7 = true;
        }

        // Check signed (-20) / 3
        let neg20 = (-20i64) as u64;
        if row[cols::N_0] == FE::from(neg20 & 0xFFFF)
            && row[cols::D_0] == FE::from(3u64)
            && row[cols::SIGNED] == FE::one()
        {
            assert_eq!(row[cols::MU_Q], FE::zero());
            assert_eq!(row[cols::MU_R], FE::one()); // wants_remainder
            // q = -6 as u64: halfwords
            let q = (-6i64) as u64;
            assert_eq!(row[cols::Q_0], FE::from(q & 0xFFFF));
            assert_eq!(row[cols::Q_1], FE::from((q >> 16) & 0xFFFF));
            assert_eq!(row[cols::Q_2], FE::from((q >> 32) & 0xFFFF));
            assert_eq!(row[cols::Q_3], FE::from((q >> 48) & 0xFFFF));
            // r = -2 as u64: halfwords
            let r = (-2i64) as u64;
            assert_eq!(row[cols::R_0], FE::from(r & 0xFFFF));
            assert_eq!(row[cols::R_3], FE::from((r >> 48) & 0xFFFF));
            // Sign flags
            assert_eq!(row[cols::SIGN_N], FE::one()); // -20 is negative
            assert_eq!(row[cols::SIGN_D], FE::zero()); // 3 is positive
            assert_eq!(row[cols::SIGN_R], FE::one()); // -2 is negative
            assert_eq!(row[cols::SIGN_Q], FE::one()); // signed && !overflow
            found_neg20_3 = true;
        }
    }

    assert!(found_100_7, "Row with n=100, d=7 not found");
    assert!(found_neg20_3, "Row with n=-20, d=3 not found");
}

#[test]
fn test_multiplicity_aggregation() {
    // Same (n, d, signed) appears multiple times with different wants_remainder flags
    let ops = vec![
        (DvrmOperation::new(20, 3, UNSIGNED), false), // wants_q
        (DvrmOperation::new(20, 3, UNSIGNED), false), // wants_q again
        (DvrmOperation::new(20, 3, UNSIGNED), true),  // wants_r
        (DvrmOperation::new(100, 7, UNSIGNED), true), // different op
    ];

    let trace = generate_dvrm_trace(&ops);

    // 2 unique rows, padded to 4
    assert_eq!(trace.main_table.height, 4);

    let mut found_20_3 = false;
    let mut found_100_7 = false;

    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::N_0] == FE::from(20u64) && row[cols::D_0] == FE::from(3u64) {
            assert_eq!(
                row[cols::MU_Q],
                FE::from(2u64),
                "Expected mu_q=2 for (20, 3)"
            );
            assert_eq!(
                row[cols::MU_R],
                FE::from(1u64),
                "Expected mu_r=1 for (20, 3)"
            );
            found_20_3 = true;
        }
        if row[cols::N_0] == FE::from(100u64) && row[cols::D_0] == FE::from(7u64) {
            assert_eq!(row[cols::MU_Q], FE::zero(), "Expected mu_q=0 for (100, 7)");
            assert_eq!(row[cols::MU_R], FE::one(), "Expected mu_r=1 for (100, 7)");
            found_100_7 = true;
        }
    }

    assert!(found_20_3, "Row with n=20, d=3 not found");
    assert!(found_100_7, "Row with n=100, d=7 not found");
}

#[test]
fn test_different_signed_flags_separate_rows() {
    // Same n/d but different signed flag should be separate rows
    let ops = vec![
        (DvrmOperation::new(20, 3, UNSIGNED), false),
        (DvrmOperation::new(20, 3, SIGNED), false),
    ];

    let trace = generate_dvrm_trace(&ops);

    // 2 unique operations, padded to 4
    assert_eq!(trace.main_table.height, 4);

    let mut count_20_3 = 0;
    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::N_0] == FE::from(20u64) && row[cols::D_0] == FE::from(3u64) {
            assert_eq!(row[cols::MU_Q], FE::one());
            count_20_3 += 1;
        }
    }

    assert_eq!(
        count_20_3, 2,
        "Should have 2 separate rows for (20, 3) with different signed flags"
    );
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();
    // Expected interactions:
    // - 12x IS_HALF senders (r×4, n_sub_r×4, q×4) — n and d are assumptions (A1, A2)
    // - 3x MSB16 senders (sign_n, sign_r, sign_d)
    // - 1x LT sender (|r| < |d|)
    // - 2x MUL senders (n_sub_r = d*q lo + hi)
    // - 6x ZERO senders (C3×2 NEG r, C5×2 NEG d, C8 overflow, C17 div_by_zero)
    // - 2x DVRM receivers (quotient, remainder)
    // Total: 12 + 3 + 1 + 2 + 6 + 2 = 26
    assert_eq!(interactions.len(), 26, "Expected 26 bus interactions");
}

#[test]
fn test_trace_div_by_zero_columns() {
    let ops = vec![(DvrmOperation::new(42, 0, UNSIGNED), false)];

    let trace = generate_dvrm_trace(&ops);

    // Find the row (non-padding)
    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::N_0] == FE::from(42u64) {
            assert_eq!(row[cols::DIV_BY_ZERO], FE::one());
            assert_eq!(row[cols::OVERFLOW], FE::zero());
            // q = all 1s: each halfword = 65535
            assert_eq!(row[cols::Q_0], FE::from(0xFFFFu64));
            assert_eq!(row[cols::Q_1], FE::from(0xFFFFu64));
            assert_eq!(row[cols::Q_2], FE::from(0xFFFFu64));
            assert_eq!(row[cols::Q_3], FE::from(0xFFFFu64));
            // r = n = 42
            assert_eq!(row[cols::R_0], FE::from(42u64));
            assert_eq!(row[cols::R_1], FE::zero());
            return;
        }
    }
    panic!("Row with n=42 not found");
}

#[test]
fn test_trace_overflow_columns() {
    let ops = vec![(DvrmOperation::new(i64::MIN as u64, u64::MAX, SIGNED), false)];

    let trace = generate_dvrm_trace(&ops);

    let n = i64::MIN as u64;
    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::N_0] == FE::from(n & 0xFFFF) && row[cols::SIGNED] == FE::one() {
            assert_eq!(row[cols::DIV_BY_ZERO], FE::zero());
            assert_eq!(row[cols::OVERFLOW], FE::one());
            // q = MIN (same as n)
            assert_eq!(row[cols::Q_0], FE::from(n & 0xFFFF));
            assert_eq!(row[cols::Q_3], FE::from((n >> 48) & 0xFFFF));
            // r = 0
            assert_eq!(row[cols::R_0], FE::zero());
            assert_eq!(row[cols::R_1], FE::zero());
            assert_eq!(row[cols::R_2], FE::zero());
            assert_eq!(row[cols::R_3], FE::zero());
            // sign_q = signed && !overflow = false
            assert_eq!(row[cols::SIGN_Q], FE::zero());
            return;
        }
    }
    panic!("Overflow row not found");
}

#[test]
fn test_zero_numerator() {
    // 0 / x = 0 remainder 0
    let op1 = DvrmOperation::new(0, 12345, UNSIGNED);
    assert_eq!(op1.compute_quotient(), 0);
    assert_eq!(op1.compute_remainder(), 0);

    let op2 = DvrmOperation::new(0, 12345, SIGNED);
    assert_eq!(op2.compute_quotient(), 0);
    assert_eq!(op2.compute_remainder(), 0);
}

#[test]
fn test_identity_division() {
    // x / 1 = x remainder 0
    let op = DvrmOperation::new(12345, 1, UNSIGNED);
    assert_eq!(op.compute_quotient(), 12345);
    assert_eq!(op.compute_remainder(), 0);
}

#[test]
fn test_padding_row() {
    // Empty operations → should still produce a valid trace with padding
    let ops: Vec<(DvrmOperation, bool)> = vec![];
    let trace = generate_dvrm_trace(&ops);

    // Minimum 4 rows (all padding: n=0, d=0, signed=false)
    assert_eq!(trace.main_table.height, 4);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    // Padding row should have all zeros (0 / 0 unsigned → q=MAX, r=0)
    // But padding has mu_q=0, mu_r=0 so the div_by_zero q=MAX values
    // don't affect bus interactions
    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::N_0], FE::zero());
    assert_eq!(row[cols::D_0], FE::zero());
    assert_eq!(row[cols::SIGNED], FE::zero());
    assert_eq!(row[cols::MU_Q], FE::zero());
    assert_eq!(row[cols::MU_R], FE::zero());
}
