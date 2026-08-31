use crate::tables::dma_set::{DmaSetOperation, cols, generate_dma_set_trace};
use crate::tables::types::FE;
use crate::test_utils::{busless_air, validate_busless};

fn row(count: u64, first: bool, end: bool) -> DmaSetOperation {
    DmaSetOperation {
        timestamp: 100,
        dst: 0x2000,
        count,
        fill: 0x3C,
        first,
        end,
    }
}

#[test]
fn dma_set_trace_uses_eight_byte_rows_then_a_byte_tail() {
    let trace = generate_dma_set_trace(&[
        row(10, true, false),
        row(2, false, false),
        row(1, false, false),
        row(0, false, true),
    ]);

    let wide = trace.main_table.get_row(0);
    assert_eq!(wide[cols::TAIL], FE::zero());
    assert_eq!(wide[cols::DST_INCR_0], FE::from(0x2008u64));
    assert_eq!(wide[cols::COUNT_DECR_0], FE::from(2u64));
    assert_eq!(wide[cols::FILL], FE::from(0x3Cu64));
    assert_eq!(wide[cols::FILL_WIDE], FE::from(0x3Cu64));

    let tail = trace.main_table.get_row(1);
    assert_eq!(tail[cols::TAIL], FE::one());
    assert_eq!(tail[cols::DST_INCR_0], FE::from(0x2001u64));
    assert_eq!(tail[cols::COUNT_DECR_0], FE::one());
    assert_eq!(tail[cols::FILL], FE::from(0x3Cu64));
    // The write tuple broadcasts FILL_WIDE into lanes 1..7, so a one-byte row
    // must zero it or the MEMW write widens past the byte it is allowed to touch.
    assert_eq!(tail[cols::FILL_WIDE], FE::zero());

    let terminal = trace.main_table.get_row(3);
    assert_eq!(terminal[cols::END], FE::one());
    assert_eq!(terminal[cols::TAIL], FE::one());
    assert_eq!(terminal[cols::COUNT_DECR_0], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_1], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_2], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_3], FE::from(0xFFFFu64));
}

#[test]
fn empty_dma_set_call_is_a_single_first_and_terminal_row() {
    let trace = generate_dma_set_trace(&[row(0, true, true)]);
    let first = trace.main_table.get_row(0);
    assert_eq!(first[cols::FIRST], FE::one());
    assert_eq!(first[cols::END], FE::one());
    assert_eq!(first[cols::MU], FE::one());
}

#[test]
fn dma_set_constraints_accept_valid_rows_and_reject_a_wide_tail_fill() {
    let mut trace = generate_dma_set_trace(&[
        row(2, true, false),
        row(1, false, false),
        row(0, false, true),
    ]);
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma_set::DmaSetConstraints);
    assert!(validate_busless(&air, &trace));

    // Row 0 is a one-byte row (count = 2 < 8). Constraint 9 (`tail * fill_wide`)
    // is the only thing stopping it from broadcasting the fill into lanes 1..7.
    trace.main_table.set(0, cols::FILL_WIDE, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a one-byte row must not smuggle a wide fill into lanes 1..7"
    );
}

#[test]
fn dma_set_constraints_reject_a_wide_row_whose_fill_wide_disagrees_with_fill() {
    let mut trace = generate_dma_set_trace(&[row(10, true, false), row(2, false, false)]);
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma_set::DmaSetConstraints);
    assert!(validate_busless(&air, &trace));

    // Row 0 is a wide row. Constraint 10 pins `fill_wide == fill`; without it the
    // seven high lanes could carry a different byte than lane 0.
    let fill = *trace.main_table.get(0, cols::FILL);
    trace.main_table.set(0, cols::FILL_WIDE, fill + FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "an eight-byte row must write the same byte in every lane"
    );
}

#[test]
fn dma_set_constraints_reject_active_destination_wrap() {
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma_set::DmaSetConstraints);

    let destination_wrap = generate_dma_set_trace(&[DmaSetOperation {
        timestamp: 100,
        dst: u64::MAX - 3,
        count: 8,
        fill: 0x3C,
        first: true,
        end: false,
    }]);
    assert!(
        !validate_busless(&air, &destination_wrap),
        "an active destination increment must not wrap modulo 2^64"
    );
}

#[test]
fn dma_set_terminal_row_may_wrap_unused_successor_columns() {
    let trace = generate_dma_set_trace(&[DmaSetOperation {
        timestamp: 100,
        dst: u64::MAX,
        count: 0,
        fill: 0x3C,
        first: true,
        end: true,
    }]);
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma_set::DmaSetConstraints);
    assert!(
        validate_busless(&air, &trace),
        "terminal successors are not consumed and may wrap"
    );
}

#[test]
fn dma_set_bus_interactions_count() {
    use crate::tables::dma_set::bus_interactions;
    assert_eq!(bus_interactions().len(), 19);
}

#[test]
fn dma_set_constraints_count_and_indices() {
    use crate::tables::dma_set::DmaSetConstraints;
    use stark::constraints::builder::ConstraintSet;
    let meta = DmaSetConstraints.meta();
    assert_eq!(meta.len(), 11);
    // Dense, idx-ordered.
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    // All constraints are degree 2 (no over-degree slips in a template change).
    assert_eq!(DmaSetConstraints.max_degree(), 2);
}

#[test]
fn dma_set_padding_row_cannot_claim_first_or_end() {
    // Constraint 4, `(first + end) * (1 - mu) = 0`, is the sole guard that a
    // padding row (mu = 0) cannot masquerade as the first or terminal row of a
    // fill — bitness alone accepts first = 1 or end = 1. A padding row claiming
    // `first` would forge an ECALL receive; claiming `end` would forge a
    // terminal row.
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma_set::DmaSetConstraints);
    let base = generate_dma_set_trace(&[
        row(2, true, false),
        row(1, false, false),
        row(0, false, true),
    ]);
    // Row 3 is padding: mu = 0, first = end = 0, and the trace validates.
    assert_eq!(base.main_table.get_row(3)[cols::MU], FE::zero());
    assert!(validate_busless(&air, &base));

    let mut forge_first = base.clone();
    forge_first.main_table.set(3, cols::FIRST, FE::one());
    assert!(
        !validate_busless(&air, &forge_first),
        "a padding row (mu = 0) must not claim to be a fill's first row"
    );

    let mut forge_end = base;
    forge_end.main_table.set(3, cols::END, FE::one());
    assert!(
        !validate_busless(&air, &forge_end),
        "a padding row (mu = 0) must not claim to be a fill's terminal row"
    );
}
