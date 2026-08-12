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

/// The canonical vectors, embedded from the validated oracle so the fixture
/// cannot drift from the model that generated it.
///
/// `docs/verification/dma/dma-oracle/test_oracle.py` emits this file next to the
/// richer JSON; it is anchored on libc `memmove`, CPython slice assignment, and a
/// row-level vs byte-level replay equivalence over every length 0..=256.
/// Embedding it — rather than hand-transcribing — is what makes a regenerated
/// oracle a compile-time input to these tests instead of a silent no-op.
///
/// Line format, one record per line:
///   `vector|<name>|<dst>|<src>|<count>|<data_rows>`
///   `row|<src>|<dst>|<count>|<tail 0|1>|<width>`
const CANONICAL_ROWS: &str =
    include_str!("../../../docs/verification/dma/dma-oracle/canonical_dma_rows.txt");

/// One case parsed out of the canonical row table.
struct OracleVector {
    name: String,
    src: u64,
    dst: u64,
    count: u64,
    /// Per data row: `(src, dst, count, tail, width)`.
    rows: Vec<(u64, u64, u64, bool, u64)>,
}

/// Parses [`CANONICAL_ROWS`]. Any malformed line is a panic, so a restructured
/// or truncated fixture fails loudly rather than silently matching nothing.
fn parse_canonical_vectors() -> Vec<OracleVector> {
    fn num(field: &str, line: &str) -> u64 {
        field
            .parse()
            .unwrap_or_else(|_| panic!("canonical rows: bad number {field:?} in {line:?}"))
    }

    let mut vectors: Vec<OracleVector> = Vec::new();
    let mut declared_rows: Vec<u64> = Vec::new();
    for line in CANONICAL_ROWS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('|').collect();
        match f[0] {
            "vector" => {
                assert_eq!(f.len(), 6, "canonical rows: bad vector line {line:?}");
                vectors.push(OracleVector {
                    name: f[1].to_string(),
                    dst: num(f[2], line),
                    src: num(f[3], line),
                    count: num(f[4], line),
                    rows: Vec::with_capacity(num(f[5], line) as usize),
                });
                declared_rows.push(num(f[5], line));
            }
            "row" => {
                assert_eq!(f.len(), 6, "canonical rows: bad row line {line:?}");
                let vector = vectors
                    .last_mut()
                    .expect("canonical rows: a row line appeared before any vector line");
                vector.rows.push((
                    num(f[1], line),
                    num(f[2], line),
                    num(f[3], line),
                    num(f[4], line) == 1,
                    num(f[5], line),
                ));
            }
            other => panic!("canonical rows: unknown record type {other:?}"),
        }
    }
    // The declared data-row count must match the rows that followed. An earlier
    // version compared `rows.len()` against a sum of ones — always true — so the
    // emitter's declared count was dead data and a fixture claiming 99 rows for a
    // 2-row vector stayed green.
    for (vector, declared) in vectors.iter().zip(&declared_rows) {
        assert_eq!(
            vector.rows.len() as u64,
            *declared,
            "{}: fixture declares {declared} data rows but {} followed",
            vector.name,
            vector.rows.len()
        );
    }
    vectors
}

/// The trace builder's row decomposition equals the oracle's.
///
/// This calls [`dma_ops_for_test`], which drives the real
/// `collect_dma_memcpy_ops` — the function that actually performs the greedy
/// `8-while-count>=8-then-1` split. An earlier version of this test drove
/// `generate_dma_trace` instead and was **vacuous**: that function only formats
/// an already-decomposed op list into columns, so the test asserted the
/// formatter echoed back the fixture the test itself had built. Mutating the
/// production width rule (`remaining >= 8` -> `remaining > 8`) left it green.
///
/// Acceptance criterion for any future change here: that mutation must fail this
/// test.
#[test]
fn dma_trace_matches_oracle_row_decomposition() {
    use crate::tables::trace_builder::dma_ops_for_test;

    let vectors = parse_canonical_vectors();
    assert_eq!(vectors.len(), 10, "expected all ten canonical vectors");
    // Pin the CONTENT too, not just the count: a fixture that degenerated every
    // vector to `count = 0` would still have ten entries, and every assertion in
    // the loop below would then be a no-op on zero data rows.
    let mut lengths: Vec<u64> = vectors.iter().map(|v| v.count).collect();
    lengths.sort_unstable();
    assert_eq!(
        lengths,
        vec![0, 1, 7, 8, 9, 16, 24, 24, 27, 256],
        "canonical vector lengths changed — regenerate the fixture and re-read this test"
    );
    // Derived from the lengths by the greedy rule, not hardcoded — a hand-typed
    // total is the same "declared, not derived" defect this campaign exists to
    // catch, and my first attempt at it was wrong (55 for an actual 57).
    let expected_rows: usize = lengths.iter().map(|n| (n / 8 + n % 8) as usize).sum();
    assert_eq!(
        vectors.iter().map(|v| v.rows.len()).sum::<usize>(),
        expected_rows,
        "total data rows across the fixture"
    );

    for vector in &vectors {
        // Seed the source region the same way the oracle's emitter does.
        let source: Vec<u8> = (0..vector.count).map(|i| (i * 7 + 3) as u8).collect();
        let (memw_ops, rows) =
            dma_ops_for_test(0x30, vector.dst, vector.src, vector.count, &source);

        let data_rows: Vec<_> = rows.iter().filter(|r| !r.end).collect();
        assert_eq!(
            data_rows.len(),
            vector.rows.len(),
            "{}: builder emitted {} data rows, oracle says {}",
            vector.name,
            data_rows.len(),
            vector.rows.len()
        );

        let mut covered = 0u64;
        for (row, &(src, dst, count, tail, width)) in data_rows.iter().zip(&vector.rows) {
            assert_eq!(
                (row.src, row.dst, row.count),
                (src, dst, count),
                "{}: row at offset {covered} is (src {:#x}, dst {:#x}, count {}), oracle says ({src:#x}, {dst:#x}, {count})",
                vector.name,
                row.src,
                row.dst,
                row.count
            );
            // `tail`/`width` are derived, so this is the greedy rule itself.
            assert_eq!(
                row.count < 8,
                tail,
                "{}: row at offset {covered} disagrees on tail",
                vector.name
            );
            assert_eq!(
                row.value[width as usize..],
                [0u8; 8][width as usize..],
                "{}: unused value lanes must be zero",
                vector.name
            );
            for lane in 0..width as usize {
                assert_eq!(
                    row.value[lane],
                    source[(covered + lane as u64) as usize],
                    "{}: copied byte at offset {} is wrong",
                    vector.name,
                    covered + lane as u64
                );
            }
            covered += width;
        }
        assert_eq!(
            covered, vector.count,
            "{}: rows cover {covered} bytes of {}",
            vector.name, vector.count
        );

        // Exactly one first row and one terminal row, and the terminal row lands
        // past the copied range.
        assert_eq!(
            rows.iter().filter(|r| r.first).count(),
            1,
            "{}",
            vector.name
        );
        assert_eq!(rows.iter().filter(|r| r.end).count(), 1, "{}", vector.name);
        // Position, not just count: `first` carries the Ecall receive, the three
        // register reads and the `count < MAX + 1` bound, so moving it off the head
        // row matters. Counting alone let that mutation through.
        assert!(
            rows[0].first,
            "{}: the head row must be `first`",
            vector.name
        );
        assert!(
            !rows[1..].iter().any(|r| r.first),
            "{}: no row after the head may claim `first`",
            vector.name
        );
        assert!(
            rows.iter().all(|r| r.timestamp == 0x30),
            "{}: every row of one ecall shares its timestamp",
            vector.name
        );
        let terminal = rows.last().expect("terminal row");
        assert!(terminal.end && terminal.count == 0, "{}", vector.name);
        assert_eq!(
            (terminal.src, terminal.dst),
            (vector.src + vector.count, vector.dst + vector.count),
            "{}: terminal row addresses",
            vector.name
        );

        // The MEMW payload. Asserting COUNTS here is not enough: the two phases
        // emit equal numbers of operations, so counting alone is satisfied by
        // swapping the two timestamps, by flipping `is_read`, or by a wrong
        // address or value. Each of those mutations previously survived. Assert
        // the fields, and assert the ORDERING as an ordering.
        let registers: Vec<_> = memw_ops.iter().filter(|o| o.is_register).collect();
        assert_eq!(registers.len(), 3, "{}: three register reads", vector.name);
        assert!(
            registers.iter().all(|o| o.timestamp == 0x30 && o.is_read),
            "{}: register operands are read at T",
            vector.name
        );
        assert_eq!(
            registers.iter().map(|o| o.base_address).collect::<Vec<_>>(),
            vec![20, 22, 24],
            "{}: registers are x10/x11/x12 at base 2*reg",
            vector.name
        );

        let data: Vec<_> = memw_ops.iter().filter(|o| !o.is_register).collect();
        assert_eq!(
            data.len(),
            2 * vector.rows.len(),
            "{}: one read and one write per data row",
            vector.name
        );
        let reads: Vec<_> = data.iter().filter(|o| o.is_read).collect();
        let writes: Vec<_> = data.iter().filter(|o| !o.is_read).collect();
        assert_eq!(
            (reads.len(), writes.len()),
            (vector.rows.len(), vector.rows.len()),
            "{}: is_read splits the data ops evenly",
            vector.name
        );

        // Every read strictly before every write — the property that gives an
        // overlapping copy its snapshot semantics. Stated as max(read) < min(write)
        // so swapping the two timestamps cannot satisfy it.
        if !vector.rows.is_empty() {
            let last_read = reads.iter().map(|o| o.timestamp).max().expect("reads");
            let first_write = writes.iter().map(|o| o.timestamp).min().expect("writes");
            assert!(
                last_read < first_write,
                "{}: every source read must precede every destination write \
                 (last read {last_read}, first write {first_write})",
                vector.name
            );
            assert_eq!(
                (last_read, first_write),
                (0x31, 0x32),
                "{}: reads at T+1, writes at T+2",
                vector.name
            );
        }

        // Addresses, widths and values, per phase and in offset order.
        let mut offset = 0u64;
        for (i, &(src, dst, _, _, width)) in vector.rows.iter().enumerate() {
            let read = reads[i];
            let write = writes[i];
            assert_eq!(
                (read.base_address, u64::from(read.width)),
                (src, width),
                "{}: read {i} addresses src with the row's width",
                vector.name
            );
            assert_eq!(
                (write.base_address, u64::from(write.width)),
                (dst, width),
                "{}: write {i} addresses dst with the row's width",
                vector.name
            );
            for lane in 0..width as usize {
                let expected = u32::from(source[(offset + lane as u64) as usize]);
                assert_eq!(
                    read.value[lane], expected,
                    "{}: read {i} lane {lane} is not the source byte",
                    vector.name
                );
                assert_eq!(
                    write.value[lane], expected,
                    "{}: write {i} lane {lane} does not carry the copied byte",
                    vector.name
                );
            }
            offset += width;
        }
    }
}

/// The maximum chunk is 33 rows with no tail row, derived from the constant
/// rather than from the test's own loop bound.
///
/// 256 is 8-aligned, so `count % 8 == 0` and every data row is a wide one — the
/// row-count bound the `Alu[count, 257, LT]` lookup exists to enforce. Asserting
/// The expectation below is still computed locally from `DMA_MEMCPY_MAX_BYTES`
/// rather than read out of production code: `trace_builder`'s own
/// `count / 8 + count % 8` feeds only a `Vec::with_capacity` hint and does not
/// drive the loop, so asserting against it would prove nothing. What makes this
/// test non-vacuous is that `rows` comes from the real decomposition, so the
/// greedy width rule and the terminal row are both exercised.
#[test]
fn dma_maximum_chunk_has_no_tail_row() {
    use crate::tables::dma::DMA_MEMCPY_MAX_BYTES as MAX;
    use crate::tables::trace_builder::dma_ops_for_test;

    let source: Vec<u8> = (0..MAX).map(|i| i as u8).collect();
    let (_, rows) = dma_ops_for_test(0x30, 0x1000, 0x2000, MAX, &source);

    // `trace_builder`'s own formula: data_rows = count / 8 + count % 8.
    let expected_data_rows = MAX / 8 + MAX % 8;
    assert_eq!(
        rows.len() as u64,
        expected_data_rows + 1,
        "expected {expected_data_rows} data rows plus one terminal row"
    );
    assert!(
        rows.iter().filter(|r| !r.end).all(|r| r.count >= 8),
        "no data row may be a tail row when the length is 8-aligned"
    );
    assert!(rows.last().expect("terminal").end);
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
