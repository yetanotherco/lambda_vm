//! Tests for the DVRM (Division/Remainder) table.

use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::dvrm::{
    DvrmConstraints, DvrmOperation, bus_interactions, cols, generate_dvrm_trace,
};
use crate::tables::types::FE;
use crate::test_utils::{
    busless_air, create_dvrm_air, in_chip_constraint_count, is_halfword_sender_columns,
    validate_busless,
};

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
    // - 8x IS_HALF senders for inputs (n×4, d×4) — A1/A2 now enforced, not assumed
    // - 12x IS_HALF senders (r×4, n_sub_r×4, q×4)
    // - 3x MSB16 senders (sign_n, sign_r, sign_d)
    // - 1x LT sender (|r| < |d|)
    // - 2x MUL senders (n_sub_r = d*q lo + hi)
    // - 6x ZERO senders (C3×2 NEG r, C5×2 NEG d, C8 overflow, C17 div_by_zero)
    // - 2x DVRM receivers (quotient, remainder)
    // Total: 8 + 12 + 3 + 1 + 2 + 6 + 2 = 34
    assert_eq!(interactions.len(), 34, "Expected 34 bus interactions");
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

// Div-by-zero remainder: a division-by-zero row must return the numerator as the
// remainder. This holds via the existing carry-chain / equality constraints
// (`n_sub_r + r = n`); an explicit `div_by_zero => r = n` constraint is a spec-level
// addition the spec does not mandate, so it is intentionally not added here.

/// Enforcement: on a division-by-zero row, forging `r != n` is rejected by the
/// carry-chain constraints (`n_sub_r + r = n`), evaluated in isolation over a bus-less
/// AIR — no explicit div-by-zero remainder constraint is needed.
#[test]
fn test_dvrm_rejects_false_div_by_zero_remainder() {
    let air = busless_air(cols::NUM_COLUMNS, DvrmConstraints);
    // numerator = 20, denominator = 0 => div-by-zero, honest remainder = 20.
    let mut trace = generate_dvrm_trace(&[(DvrmOperation::new(20, 0, UNSIGNED), true)]);
    assert!(
        validate_busless(&air, &trace),
        "honest div-by-zero row (r = n = 20) must validate"
    );

    trace.set_main(0, cols::R_0, FE::from(999u64));
    assert!(
        !validate_busless(&air, &trace),
        "a forged remainder on div-by-zero must be rejected by the carry-chain constraints"
    );
}

// Soundness regression (VM-5): the denominator halves must be IS_HALFWORD
// range-checked so a prover cannot forge `div_by_zero` via non-canonical halves.

/// Presence: the denominator halves are range-checked via IS_HALFWORD senders.
#[test]
fn test_dvrm_range_checks_denominator_halves() {
    let cols_checked = is_halfword_sender_columns(&bus_interactions());
    for c in [cols::D_0, cols::D_1, cols::D_2, cols::D_3] {
        assert!(
            cols_checked.contains(&c),
            "DVRM must IS_HALF range-check denominator half column {c}"
        );
    }
}

/// Wiring: `create_dvrm_air` registers its in-chip constraints on top of its bus
/// constraints. Catches a revert to `transition_constraints = vec![]` or a dropped
/// constraint.
#[test]
fn test_dvrm_air_wires_in_chip_constraints() {
    let air = create_dvrm_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        cols::NUM_COLUMNS,
        bus_interactions(),
    );
    use stark::constraints::builder::ConstraintSet;
    assert_eq!(in_chip, DvrmConstraints.meta().len());
}

/// Regression test for the `Msb16` LogUp over-send bug.
///
/// DVRM is split into chip instances of `max_rows.dvrm` raw ops (`chunk_and_generate`)
/// and each instance deduplicates only its own chunk, sending its three MSB16 sign
/// lookups once per unique signed op *per instance* (multiplicity = the `SIGNED` bit).
/// So `collect_bitwise_from_dvrm`, which feeds the BITWISE MSB16 multiplicity, must use
/// the *same* per-chunk dedup: a unique signed op spanning two instances is sent twice
/// but, with a single global dedup, would be tallied once — leaving the `Msb16` bus
/// unbalanced and verification failing for any block large enough to split DVRM.
#[test]
fn msb16_bitwise_multiplicity_matches_per_instance_sends() {
    use crate::tables::bitwise::BitwiseOperationType;
    use crate::tables::trace_builder::collect_bitwise_from_dvrm;

    let chunk = 4usize;
    // One unique signed div op repeated so it spans two `chunk`-sized instances.
    let op = DvrmOperation::new(0x0123_4567_89ab_cdef, 0x0000_0000_0001_0001, true);
    let ops: Vec<(DvrmOperation, bool)> = std::iter::repeat_n((op, false), 6).collect();
    assert!(
        ops.len() > chunk,
        "scenario must split DVRM into >1 instance"
    );

    // The DVRM AIR sends three MSB16 lookups (n[3], r[3], d[3]) each with multiplicity
    // Column(SIGNED) per row, so total sends = Σ rows (SIGNED) × 3.
    let mut sends = 0usize;
    for c in ops.chunks(chunk) {
        let trace = generate_dvrm_trace(c);
        for row in 0..trace.num_rows() {
            if *trace.get_main(row, cols::SIGNED) == FE::one() {
                sends += 3;
            }
        }
    }
    assert_eq!(sends, 6, "sanity: 2 instances × 3 MSB16 sends");

    let tallied = collect_bitwise_from_dvrm(&ops, chunk)
        .iter()
        .filter(|b| matches!(b.lookup_type, BitwiseOperationType::Msb16))
        .count();
    assert_eq!(
        tallied, sends,
        "BITWISE MSB16 multiplicity ({tallied}) must equal total DVRM-instance MSB16 \
         sends ({sends}); a mismatch leaves the Msb16 bus unbalanced"
    );
}

/// Regression test for the DVRM NEG-template ZERO lookups — the *other* per-unique
/// bit-gated loop the same fix converted to per-chunk dedup (the MSB16 test above
/// covers the first one). C3/C5 emit ZERO lookups gated by the `SIGN_R`/`SIGN_D` bits,
/// once per unique signed op, so they must deduplicate PER CHIP INSTANCE just like MSB16.
#[test]
fn neg_template_zero_lookups_dedup_per_chip_instance() {
    use crate::tables::bitwise::BitwiseOperationType;
    use crate::tables::trace_builder::collect_bitwise_from_dvrm;

    // Signed op with negative remainder AND negative divisor -> sign_r = sign_d = 1, so the
    // NEG template (C3/C5) emits per-unique ZERO lookups gated by those bits.
    let op = DvrmOperation::new((-20i64) as u64, (-3i64) as u64, SIGNED);
    assert!(
        op.sign_r() && op.sign_d(),
        "scenario needs sign_r = sign_d = 1"
    );
    let ops: Vec<(DvrmOperation, bool)> = std::iter::repeat_n((op, false), 6).collect();

    let zero_lookups = |chunk: usize| {
        collect_bitwise_from_dvrm(&ops, chunk)
            .iter()
            .filter(|b| matches!(b.lookup_type, BitwiseOperationType::Zero))
            .count()
    };

    // The per-raw ZERO lookups (C8/C20) are identical regardless of chunking; only the
    // per-unique NEG-template ZEROs differ. A global dedup emits them once for the whole
    // list, so the count would NOT change with chunk size; the per-instance fix emits them
    // once per instance, so two instances must produce strictly more ZERO lookups than one.
    // (Without the fix these are equal and this assertion fails.)
    let one_instance = zero_lookups(ops.len()); // chunks(6) -> 1 chunk
    let two_instances = zero_lookups(4); // chunks(4) -> [4],[2] -> 2 chunks
    assert!(
        one_instance > 0,
        "expected NEG-template ZERO lookups to be emitted"
    );
    assert!(
        two_instances > one_instance,
        "per-instance dedup of NEG-template ZERO lookups regressed: \
         {two_instances} (2 instances) must exceed {one_instance} (1 instance)"
    );
}
