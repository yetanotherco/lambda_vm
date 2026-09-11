use crate::tables::dma::{DmaOperation, cols, generate_dma_trace};
use crate::tables::types::FE;
use crate::test_utils::{busless_air, validate_busless};

fn row(count: u64, first: bool, end: bool, value: [u8; 8]) -> DmaOperation {
    DmaOperation {
        timestamp: 100,
        src: 0x1000,
        dst: 0x2000,
        count,
        first,
        end,
        value,
    }
}

#[test]
fn dma_trace_uses_eight_byte_rows_then_a_byte_tail() {
    let trace = generate_dma_trace(&[
        row(10, true, false, *b"abcdefgh"),
        row(2, false, false, [b'i', 0, 0, 0, 0, 0, 0, 0]),
        row(1, false, false, [b'j', 0, 0, 0, 0, 0, 0, 0]),
        row(0, false, true, [0; 8]),
    ]);

    let wide = trace.main_table.get_row(0);
    assert_eq!(wide[cols::TAIL], FE::zero());
    assert_eq!(wide[cols::SRC_INCR_0], FE::from(0x1008u64));
    assert_eq!(wide[cols::COUNT_DECR_0], FE::from(2u64));
    for (i, &byte) in b"abcdefgh".iter().enumerate() {
        assert_eq!(wide[cols::VALUE[i]], FE::from(byte as u64));
    }

    let tail = trace.main_table.get_row(1);
    assert_eq!(tail[cols::TAIL], FE::one());
    assert_eq!(tail[cols::SRC_INCR_0], FE::from(0x1001u64));
    assert_eq!(tail[cols::COUNT_DECR_0], FE::one());
    assert_eq!(tail[cols::VALUE[0]], FE::from(b'i' as u64));
    assert!(cols::VALUE[1..].iter().all(|&c| tail[c] == FE::zero()));

    let terminal = trace.main_table.get_row(3);
    assert_eq!(terminal[cols::END], FE::one());
    assert_eq!(terminal[cols::TAIL], FE::one());
    assert_eq!(terminal[cols::COUNT_DECR_0], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_1], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_2], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_3], FE::from(0xFFFFu64));
}

#[test]
fn empty_dma_call_is_a_single_first_and_terminal_row() {
    let trace = generate_dma_trace(&[row(0, true, true, [0; 8])]);
    let first = trace.main_table.get_row(0);
    assert_eq!(first[cols::FIRST], FE::one());
    assert_eq!(first[cols::END], FE::one());
    assert_eq!(first[cols::MU], FE::one());
}

#[test]
fn dma_constraints_accept_valid_rows_and_reject_nonzero_tail_lanes() {
    let mut trace = generate_dma_trace(&[
        row(2, true, false, [b'a', 0, 0, 0, 0, 0, 0, 0]),
        row(1, false, false, [b'b', 0, 0, 0, 0, 0, 0, 0]),
        row(0, false, true, [0; 8]),
    ]);
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma::DmaConstraints);
    assert!(validate_busless(&air, &trace));

    trace.main_table.set(0, cols::VALUE[1], FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a one-byte row must not smuggle additional copied lanes"
    );
}

#[test]
fn dma_constraints_reject_active_source_or_destination_wrap() {
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma::DmaConstraints);

    let source_wrap = generate_dma_trace(&[DmaOperation {
        timestamp: 100,
        src: u64::MAX - 3,
        dst: 0x2000,
        count: 8,
        first: true,
        end: false,
        value: [0; 8],
    }]);
    assert!(
        !validate_busless(&air, &source_wrap),
        "an active source increment must not wrap modulo 2^64"
    );

    let destination_wrap = generate_dma_trace(&[DmaOperation {
        timestamp: 100,
        src: 0x1000,
        dst: u64::MAX - 3,
        count: 8,
        first: true,
        end: false,
        value: [0; 8],
    }]);
    assert!(
        !validate_busless(&air, &destination_wrap),
        "an active destination increment must not wrap modulo 2^64"
    );
}

#[test]
fn dma_terminal_row_may_wrap_unused_successor_columns() {
    let trace = generate_dma_trace(&[DmaOperation {
        timestamp: 100,
        src: u64::MAX,
        dst: u64::MAX,
        count: 0,
        first: true,
        end: true,
        value: [0; 8],
    }]);
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma::DmaConstraints);
    assert!(
        validate_busless(&air, &trace),
        "terminal successors are not consumed and may wrap"
    );
}

#[test]
fn dma_bus_interactions_count() {
    use crate::tables::dma::bus_interactions;
    assert_eq!(bus_interactions().len(), 23);
}

#[test]
fn dma_constraints_count_and_indices() {
    use crate::tables::dma::DmaConstraints;
    use stark::constraints::builder::ConstraintSet;
    let meta = DmaConstraints.meta();
    assert_eq!(meta.len(), 18);
    // Dense, idx-ordered.
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    // All constraints are degree 2 (no over-degree slips in a template change).
    assert_eq!(DmaConstraints.max_degree(), 2);
}

#[test]
fn dma_padding_row_cannot_claim_first_or_end() {
    // Constraint 4, `(first + end) * (1 - mu) = 0`, is the sole guard that a
    // padding row (mu = 0) cannot masquerade as the first or terminal row of a
    // copy — bitness alone accepts first = 1 or end = 1, so nothing else rejects
    // it. A padding row claiming `first` would forge an ECALL receive; claiming
    // `end` would forge a copy's terminal row.
    let air = busless_air(cols::NUM_COLUMNS, crate::tables::dma::DmaConstraints);
    let base = generate_dma_trace(&[
        row(2, true, false, [b'a', 0, 0, 0, 0, 0, 0, 0]),
        row(1, false, false, [b'b', 0, 0, 0, 0, 0, 0, 0]),
        row(0, false, true, [0; 8]),
    ]);
    // Row 3 is padding: mu = 0, first = end = 0, and the trace validates.
    assert_eq!(base.main_table.get_row(3)[cols::MU], FE::zero());
    assert!(validate_busless(&air, &base));

    let mut forge_first = base.clone();
    forge_first.main_table.set(3, cols::FIRST, FE::one());
    assert!(
        !validate_busless(&air, &forge_first),
        "a padding row (mu = 0) must not claim to be a copy's first row"
    );

    let mut forge_end = base;
    forge_end.main_table.set(3, cols::END, FE::one());
    assert!(
        !validate_busless(&air, &forge_end),
        "a padding row (mu = 0) must not claim to be a copy's terminal row"
    );
}
