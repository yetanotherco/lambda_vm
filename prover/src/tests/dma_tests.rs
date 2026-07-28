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
