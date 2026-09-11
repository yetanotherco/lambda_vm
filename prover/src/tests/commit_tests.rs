//! Tests for the COMMIT (ECALL) table.
//!
//! COMMIT is now one row per `sys_write` ECALL: it accepts the syscall number, reads
//! the operand registers, advances the committed-length register x254, and hands the
//! byte loop to MEMMOVE over `BusId::CommitDefer`. Everything that modelled a
//! per-byte sequence — `first`, `end`, `value`, `address_incr`, `count_decr` and
//! their range checks — went with the loop, so the tests that exercised that
//! machinery went with it too. What is left is the row shape, the padding, and the
//! interaction and constraint inventory.

use crate::tables::commit::{CommitOperation, cols, generate_commit_trace};
use crate::tables::types::{FE, VmTable};
use crate::test_utils::{busless_air, validate_busless};

fn op(timestamp: u64, index: u64, address: u64, count: u64) -> CommitOperation {
    CommitOperation {
        timestamp,
        index,
        address,
        count,
    }
}

// =========================================================================
// Trace generation
// =========================================================================

#[test]
fn a_commit_ecall_is_one_row_carrying_its_operands() {
    let trace = generate_commit_trace(&[op(0x1234_5678_9ABC, 42, 0x2000, 7)]);
    let r = trace.main_table.get_row(0);

    assert_eq!(r[cols::TIMESTAMP_0], FE::from(0x5678_9ABCu64));
    assert_eq!(r[cols::TIMESTAMP_1], FE::from(0x1234u64));
    assert_eq!(r[cols::INDEX], FE::from(42u64));
    assert_eq!(r[cols::ADDRESS_0], FE::from(0x2000u64));
    assert_eq!(r[cols::ADDRESS_1], FE::zero());
    assert_eq!(r[cols::COUNT_0], FE::from(7u64));
    assert_eq!(r[cols::COUNT_1], FE::zero());
    assert_eq!(r[cols::MU], FE::one());
}

#[test]
fn several_ecalls_are_several_rows_and_nothing_chains_them() {
    // Two commits: 3 bytes from index 0, then 9 bytes from index 3. Under the old
    // per-byte design this was 3 + 1 + 9 + 1 rows linked by CommitNextByte; it is now
    // exactly two independent rows.
    let trace = generate_commit_trace(&[op(100, 0, 0x2000, 3), op(200, 3, 0x3000, 9)]);

    let first = trace.main_table.get_row(0);
    assert_eq!(first[cols::INDEX], FE::zero());
    assert_eq!(first[cols::COUNT_0], FE::from(3u64));

    let second = trace.main_table.get_row(1);
    assert_eq!(second[cols::TIMESTAMP_0], FE::from(200u64));
    assert_eq!(second[cols::INDEX], FE::from(3u64));
    assert_eq!(second[cols::COUNT_0], FE::from(9u64));
}

#[test]
fn a_zero_length_commit_is_still_a_row() {
    // The ECALL happened and x254 still has to be read and written, so the row exists
    // even though MEMMOVE will copy nothing.
    let trace = generate_commit_trace(&[op(100, 5, 0x2000, 0)]);
    let r = trace.main_table.get_row(0);
    assert_eq!(r[cols::COUNT_0], FE::zero());
    assert_eq!(r[cols::MU], FE::one());
}

#[test]
fn padding_rows_are_all_zero() {
    // The ADD/SUB templates that forced a non-zero padding row are gone with
    // `address_incr` and `count_decr`, so padding is plain zero now.
    let trace = generate_commit_trace(&[op(100, 0, 0x2000, 1)]);
    assert_eq!(trace.num_rows(), 4);
    for row_idx in 1..4 {
        let r = trace.main_table.get_row(row_idx);
        for (col, value) in r.iter().enumerate().take(cols::NUM_COLUMNS) {
            assert_eq!(*value, FE::zero(), "padding row {row_idx}, column {col}");
        }
    }
}

#[test]
fn the_table_pads_to_a_power_of_two_with_a_floor_of_four() {
    assert_eq!(generate_commit_trace(&[]).num_rows(), 4);
    assert_eq!(generate_commit_trace(&[op(1, 0, 0x2000, 1)]).num_rows(), 4);
    let five: Vec<_> = (0..5).map(|i| op(i, i, 0x2000, 1)).collect();
    assert_eq!(generate_commit_trace(&five).num_rows(), 8);
    assert_eq!(
        generate_commit_trace(&[op(1, 0, 0x2000, 1)])
            .main_table
            .get_row(0)
            .len(),
        cols::NUM_COLUMNS
    );
}

#[test]
fn a_full_width_timestamp_survives_the_limb_split() {
    let trace = generate_commit_trace(&[op(u64::MAX, 0, 0x2000, 1)]);
    let r = trace.main_table.get_row(0);
    assert_eq!(r[cols::TIMESTAMP_0], FE::from(0xFFFF_FFFFu64));
    assert_eq!(r[cols::TIMESTAMP_1], FE::from(0xFFFF_FFFFu64));
}

// =========================================================================
// Constraints
// =========================================================================

#[test]
fn mu_must_be_a_bit_and_that_is_the_only_constraint() {
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::commit::CommitConstraints);

    let honest = generate_commit_trace(&[op(100, 0, 0x2000, 4)]);
    assert!(validate_busless(&air, &honest), "an honest row must pass");

    let mut forged = honest.clone();
    forged.main_table.set_fe(0, cols::MU, FE::from(2u64));
    assert!(
        !validate_busless(&air, &forged),
        "mu must be constrained to a bit"
    );
}

// =========================================================================
// Inventory — these pin the deferral, so a regression shows up here first
// =========================================================================

#[test]
fn test_bus_interactions_count() {
    use crate::tables::commit::bus_interactions;
    // Ecall receive, CommitDefer send, and four register accesses (x10 read+write,
    // x11 read, x12 read, x254 read+write). The eight IsHalfword range checks and the
    // Zero end-detection went with the byte loop.
    assert_eq!(bus_interactions().len(), 6);
    // Every one of them now rides `mu`: with one row per ECALL, `first` was
    // identically `mu` and the column is gone.
    use stark::lookup::Multiplicity;
    for (i, interaction) in bus_interactions().iter().enumerate() {
        assert!(
            matches!(interaction.multiplicity, Multiplicity::Column(c) if c == cols::MU),
            "interaction {i} should ride mu"
        );
    }
}

#[test]
fn test_constraints_count_and_indices() {
    use crate::tables::commit::CommitConstraints;
    use stark::constraints::builder::ConstraintSet;
    let meta = CommitConstraints.meta();
    assert_eq!(meta.len(), 1);
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    assert_eq!(CommitConstraints.max_degree(), 2);
}
