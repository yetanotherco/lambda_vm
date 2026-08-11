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

/// One pinned case from the validated oracle (`docs/verification/dma/dma-oracle/`).
struct OracleVector {
    name: &'static str,
    src: u64,
    dst: u64,
    count: u64,
    /// Per data row: `(src, dst, count, tail, width)`.
    rows: &'static [(u64, u64, u64, bool, u64)],
}

/// Structural cases from `docs/verification/dma/dma-oracle/canonical_dma_vectors.json`,
/// which the oracle harness regenerates and which is anchored on libc `memmove`
/// plus a row-level/byte-level replay equivalence. Do not edit by hand: rerun
/// `python3 docs/verification/dma/dma-oracle/test_oracle.py` and re-transcribe.
///
/// `maximum chunk` (n = 256) is abbreviated to its first and last data rows —
/// it is the only case with no tail row at all, which is the property worth
/// pinning; `widest tail` (n = 7) is the opposite extreme, eight rows to move
/// seven bytes.
const ORACLE_VECTORS: &[OracleVector] = &[
    OracleVector {
        name: "empty",
        src: 0x2000,
        dst: 0x1000,
        count: 0,
        rows: &[],
    },
    OracleVector {
        name: "single byte",
        src: 0x2000,
        dst: 0x1000,
        count: 1,
        rows: &[(0x2000, 0x1000, 1, true, 1)],
    },
    OracleVector {
        name: "one wide row",
        src: 0x2000,
        dst: 0x1000,
        count: 8,
        rows: &[(0x2000, 0x1000, 8, false, 8)],
    },
    OracleVector {
        name: "wide plus tail",
        src: 0x2000,
        dst: 0x1000,
        count: 9,
        rows: &[(0x2000, 0x1000, 9, false, 8), (0x2008, 0x1008, 1, true, 1)],
    },
    OracleVector {
        name: "widest tail",
        src: 0x2000,
        dst: 0x1000,
        count: 7,
        rows: &[
            (0x2000, 0x1000, 7, true, 1),
            (0x2001, 0x1001, 6, true, 1),
            (0x2002, 0x1002, 5, true, 1),
            (0x2003, 0x1003, 4, true, 1),
            (0x2004, 0x1004, 3, true, 1),
            (0x2005, 0x1005, 2, true, 1),
            (0x2006, 0x1006, 1, true, 1),
        ],
    },
    OracleVector {
        name: "unaligned body and tail",
        src: 0x1003,
        dst: 0x2005,
        count: 27,
        rows: &[
            (0x1003, 0x2005, 27, false, 8),
            (0x100B, 0x200D, 19, false, 8),
            (0x1013, 0x2015, 11, false, 8),
            (0x101B, 0x201D, 3, true, 1),
            (0x101C, 0x201E, 2, true, 1),
            (0x101D, 0x201F, 1, true, 1),
        ],
    },
    OracleVector {
        name: "page crossing",
        src: 0x1FFC,
        dst: 0x0FFC,
        count: 16,
        rows: &[
            (0x1FFC, 0x0FFC, 16, false, 8),
            (0x2004, 0x1004, 8, false, 8),
        ],
    },
];

/// The trace generator's row decomposition equals the oracle's.
///
/// The AIR proves that a row sequence is internally consistent — each row's
/// successor is its own `src + width`, the chain terminates when `count` hits
/// zero. It cannot prove the sequence is the one the *executor* performed; that
/// is what the memory bus does, and what this test does at the row level.
/// Without it the AIR and the trace builder could agree on a decomposition that
/// is self-consistent and wrong (every mutant in the oracle's own sweep is
/// exactly that shape).
#[test]
fn dma_trace_matches_oracle_row_decomposition() {
    for vector in ORACLE_VECTORS {
        // Rebuild the operations the trace builder would emit for this case.
        let mut ops: Vec<DmaOperation> = vector
            .rows
            .iter()
            .enumerate()
            .map(|(i, &(src, dst, count, _, width))| DmaOperation {
                timestamp: 0x30,
                src,
                dst,
                count,
                first: i == 0,
                end: false,
                value: {
                    let mut value = [0u8; 8];
                    for (lane, slot) in value.iter_mut().enumerate().take(width as usize) {
                        *slot = (src as u8).wrapping_add(lane as u8);
                    }
                    value
                },
            })
            .collect();
        ops.push(DmaOperation {
            timestamp: 0x30,
            src: vector.src + vector.count,
            dst: vector.dst + vector.count,
            count: 0,
            first: vector.rows.is_empty(),
            end: true,
            value: [0; 8],
        });

        let trace = generate_dma_trace(&ops);

        // The widths sum to `count`, so the copy covers the whole range once.
        let covered: u64 = vector.rows.iter().map(|&(.., width)| width).sum();
        assert_eq!(
            covered, vector.count,
            "{}: rows cover {covered} bytes, expected {}",
            vector.name, vector.count
        );

        for (i, &(src, dst, count, tail, width)) in vector.rows.iter().enumerate() {
            let row = trace.main_table.get_row(i);
            assert_eq!(
                row[cols::SRC_0],
                FE::from(src),
                "{}: row {i} src",
                vector.name
            );
            assert_eq!(
                row[cols::DST_0],
                FE::from(dst),
                "{}: row {i} dst",
                vector.name
            );
            assert_eq!(
                row[cols::COUNT_0],
                FE::from(count),
                "{}: row {i} count",
                vector.name
            );
            assert_eq!(
                row[cols::TAIL],
                if tail { FE::one() } else { FE::zero() },
                "{}: row {i} tail — the LT lookup pins this, so a mismatch here \
                 means the trace builder and the AIR disagree on the width",
                vector.name
            );
            // `src_incr`/`dst_incr` are DWordHL: the low halfword suffices for
            // these addresses, and the whole point is that it equals src + width.
            assert_eq!(
                row[cols::SRC_INCR_0],
                FE::from((src + width) & 0xFFFF),
                "{}: row {i} src_incr",
                vector.name
            );
            assert_eq!(
                row[cols::COUNT_DECR_0],
                FE::from(count - width),
                "{}: row {i} count_decr",
                vector.name
            );
            assert_eq!(
                row[cols::MU],
                FE::one(),
                "{}: row {i} must be active",
                vector.name
            );
        }

        // The terminal row: count zero, end set, all four count_decr halfwords
        // 0xFFFF (which is what the Zero-bus end detection reads).
        let terminal = trace.main_table.get_row(vector.rows.len());
        assert_eq!(
            terminal[cols::END],
            FE::one(),
            "{}: terminal end",
            vector.name
        );
        assert_eq!(
            terminal[cols::COUNT_0],
            FE::zero(),
            "{}: terminal count",
            vector.name
        );
        for lane in 0..4 {
            assert_eq!(
                terminal[cols::COUNT_DECR_0 + lane],
                FE::from(0xFFFFu64),
                "{}: terminal count_decr[{lane}]",
                vector.name
            );
        }
        assert_eq!(
            terminal[cols::FIRST],
            if vector.rows.is_empty() {
                FE::one()
            } else {
                FE::zero()
            },
            "{}: an empty copy's single row is both first and terminal",
            vector.name
        );
    }
}

/// The maximum chunk has no tail row: 256 is 8-aligned, so 32 wide rows plus a
/// terminal, and every `count_decr` is a clean multiple of eight.
///
/// Pinned separately because it is the row-count bound the `Alu[count, 257, LT]`
/// lookup exists to enforce — 33 rows is the most one guest instruction can add
/// to a continuation epoch.
#[test]
fn dma_maximum_chunk_is_thirty_three_rows_with_no_tail() {
    use crate::tables::dma::DMA_MEMCPY_MAX_BYTES;

    let count = DMA_MEMCPY_MAX_BYTES;
    let mut ops: Vec<DmaOperation> = (0..count / 8)
        .map(|k| DmaOperation {
            timestamp: 0x30,
            src: 0x2000 + k * 8,
            dst: 0x1000 + k * 8,
            count: count - k * 8,
            first: k == 0,
            end: false,
            value: [k as u8; 8],
        })
        .collect();
    ops.push(DmaOperation {
        timestamp: 0x30,
        src: 0x2000 + count,
        dst: 0x1000 + count,
        count: 0,
        first: false,
        end: true,
        value: [0; 8],
    });

    assert_eq!(
        ops.len(),
        33,
        "256 bytes is 32 wide rows plus the terminal row"
    );
    let trace = generate_dma_trace(&ops);
    for row_idx in 0..32 {
        let row = trace.main_table.get_row(row_idx);
        assert_eq!(
            row[cols::TAIL],
            FE::zero(),
            "row {row_idx} must be a wide row"
        );
        assert_eq!(
            row[cols::COUNT_DECR_0],
            FE::from(count - (row_idx as u64 + 1) * 8)
        );
    }
    assert_eq!(trace.main_table.get_row(32)[cols::END], FE::one());
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
