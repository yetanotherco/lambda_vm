use crate::tables::memmove::{MemmoveOperation, cols, generate_memmove_trace};
use crate::tables::types::{FE, VmTable};
use crate::test_utils::{busless_air, validate_busless};

/// A memset row. The contract the AIR pins is `dst = src + 8`, so build it that
/// way by default and let the tests below break it deliberately.
fn set_row(count: u64, first: bool, end: bool, src: u64, dst: u64) -> MemmoveOperation {
    MemmoveOperation {
        width: if count < 8 { 1 } else { 8 },
        functionality: crate::tables::memmove::Functionality::Set,
        timestamp: 100,
        src,
        dst,
        count,
        first,
        end,
        // A narrow row must leave lanes 1..7 clear (constraints 25-31), and the
        // terminal row copies nothing at all.
        value: if end {
            [0; 8]
        } else if count < 8 {
            [0xAB, 0, 0, 0, 0, 0, 0, 0]
        } else {
            [0xAB; 8]
        },
    }
}

/// A row whose width is chosen independently of `count`, so a test can express
/// `tail != (count < 8)`. `row()` and `set_row()` both derive width from count, which
/// makes `(1 - tail) * lt8` identically zero and constraint 14 impossible to state.
fn row_of_width(
    functionality: crate::tables::memmove::Functionality,
    count: u64,
    width: u8,
    first: bool,
    end: bool,
    src: u64,
    dst: u64,
) -> MemmoveOperation {
    MemmoveOperation {
        width,
        functionality,
        timestamp: 100,
        src,
        dst,
        count,
        first,
        end,
        value: if width == 1 {
            [0xAB, 0, 0, 0, 0, 0, 0, 0]
        } else {
            [0xAB; 8]
        },
    }
}

fn row(count: u64, first: bool, end: bool, value: [u8; 8]) -> MemmoveOperation {
    MemmoveOperation {
        width: if count < 8 { 1 } else { 8 },
        functionality: crate::tables::memmove::Functionality::Copy,
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
fn memmove_trace_uses_eight_byte_rows_then_a_byte_tail() {
    let trace = generate_memmove_trace(&[
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
    assert_eq!(terminal[cols::COUNT_DECR_0 + 1], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_0 + 2], FE::from(0xFFFFu64));
    assert_eq!(terminal[cols::COUNT_DECR_0 + 3], FE::from(0xFFFFu64));
}

#[test]
fn empty_memmove_call_is_a_single_first_and_terminal_row() {
    let trace = generate_memmove_trace(&[row(0, true, true, [0; 8])]);
    let first = trace.main_table.get_row(0);
    assert_eq!(first[cols::FIRST], FE::one());
    assert_eq!(first[cols::END], FE::one());
    assert_eq!(first[cols::MU], FE::one());
}

#[test]
fn memmove_constraints_accept_valid_rows_and_reject_nonzero_tail_lanes() {
    let mut trace = generate_memmove_trace(&[
        row(2, true, false, [b'a', 0, 0, 0, 0, 0, 0, 0]),
        row(1, false, false, [b'b', 0, 0, 0, 0, 0, 0, 0]),
        row(0, false, true, [0; 8]),
    ]);
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );
    assert!(validate_busless(&air, &trace));

    trace.main_table.set(0, cols::VALUE[1], FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a one-byte row must not smuggle additional copied lanes"
    );
}

#[test]
fn memmove_constraints_reject_active_source_or_destination_wrap() {
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );

    let source_wrap = generate_memmove_trace(&[MemmoveOperation {
        width: 8,
        functionality: crate::tables::memmove::Functionality::Copy,
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

    let destination_wrap = generate_memmove_trace(&[MemmoveOperation {
        width: 8,
        functionality: crate::tables::memmove::Functionality::Copy,
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
fn memmove_terminal_row_may_wrap_unused_successor_columns() {
    let trace = generate_memmove_trace(&[MemmoveOperation {
        width: 1,
        functionality: crate::tables::memmove::Functionality::Copy,
        timestamp: 100,
        src: u64::MAX,
        dst: u64::MAX,
        count: 0,
        first: true,
        end: true,
        value: [0; 8],
    }]);
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );
    assert!(
        validate_busless(&air, &trace),
        "terminal successors are not consumed and may wrap"
    );
}

#[test]
fn memmove_bus_interactions_count() {
    use crate::tables::memmove::bus_interactions;
    // 23 on the DMA table this replaces, plus the CommitDefer receive and eight
    // COMMIT-domain sends — one `(index, value)` pair per byte, lane 0 at `mu_com`
    // and lanes 1..7 at `mu_com_wide`. One tuple per row would be two sends and
    // three fewer aux columns (the count is `ceil(interactions / 2)`), but it would
    // make the verifier's rebuild depend on the prover's row schedule, which
    // restarts at every commit ECALL while the verifier sees only the concatenated
    // `public_output`. Per-byte pairs are what make the two sides agree.
    assert_eq!(bus_interactions().len(), 32);
}

#[test]
fn memmove_constraints_count_and_indices() {
    use crate::tables::memmove::MemmoveConstraints;
    use stark::constraints::builder::ConstraintSet;
    let meta = MemmoveConstraints.meta();
    assert_eq!(meta.len(), 34);
    // Dense, idx-ordered.
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    // All constraints are degree 2 (no over-degree slips in a template change).
    assert_eq!(MemmoveConstraints.max_degree(), 2);
}

#[test]
fn memmove_padding_row_cannot_claim_first_or_end() {
    // Constraint 4, `(first + end) * (1 - mu) = 0`, is the sole guard that a
    // padding row (mu = 0) cannot masquerade as the first or terminal row of a
    // copy — bitness alone accepts first = 1 or end = 1, so nothing else rejects
    // it. A padding row claiming `first` would forge an ECALL receive; claiming
    // `end` would forge a copy's terminal row.
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );
    let base = generate_memmove_trace(&[
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

/// Pins the shape of the COMMIT-domain sends: eight `(index, value)` pairs, one per
/// byte, indexed off `dst` — not one eight-lane tuple per row.
///
/// The verifier rebuilds this bus from `public_output` alone and never learns where
/// one commit ECALL ended and the next began. Per-byte pairs are what make its
/// rebuild independent of the prover's row schedule, which restarts at every ECALL;
/// `test_prover_tuples_are_independent_of_the_ecall_split` is the arithmetic half of
/// the same argument, and this is the half that keeps it anchored to the real chip.
#[test]
fn memmove_commit_sends_one_pair_per_byte() {
    use crate::tables::memmove::bus_interactions;
    use crate::tables::types::BusId;
    use stark::lookup::{BusValue, LinearTerm, Multiplicity, Packing};

    let commit_sends: Vec<_> = bus_interactions()
        .into_iter()
        .filter(|interaction| interaction.bus_id == BusId::Commit as u64)
        .collect();

    assert_eq!(commit_sends.len(), 8, "one COMMIT send per byte lane");

    for (lane, interaction) in commit_sends.iter().enumerate() {
        assert!(interaction.is_sender, "lane {lane} must send");

        // Lane 0 rides every copying row; lanes 1..7 only an eight-byte row, so a
        // one-byte row sends no spurious `(index, 0)` pairs.
        let expected_multiplicity = if lane == 0 {
            cols::MU_COM
        } else {
            cols::MU_COM_WIDE
        };
        assert!(
            matches!(interaction.multiplicity, Multiplicity::Column(column) if column == expected_multiplicity),
            "lane {lane} has the wrong multiplicity"
        );

        assert_eq!(
            interaction.values.len(),
            2,
            "lane {lane} is an (index, value) pair"
        );

        match &interaction.values[0] {
            BusValue::Linear(terms) => {
                assert!(
                    matches!(
                        terms.as_slice(),
                        [
                            LinearTerm::Column { coefficient: 1, column },
                            LinearTerm::Constant(offset),
                        ] if *column == cols::DST_0 && *offset == lane as i64
                    ),
                    "lane {lane} must be indexed at dst + {lane}"
                );
            }
            _ => panic!("lane {lane}'s index must be a linear combination"),
        }

        assert!(
            matches!(
                interaction.values[1],
                BusValue::Packed {
                    start_column,
                    packing: Packing::Direct,
                } if start_column == cols::VALUE[lane]
            ),
            "lane {lane} must carry value[{lane}]"
        );
    }
}

/// The memset operand contract, in the direction that matters.
///
/// `dst == src` is the degenerate case: read and write address the same cell at
/// adjacent timestamps, so the memory argument closes on `value == value` and every
/// value lane becomes a free field element. Constraints 32 and 33 are the only thing
/// that rejects it, so both are tested here in both directions, and a `Copy` row is
/// tested to confirm the gate is `is_set` and does not leak onto the copy path (a
/// memcpy with `dst == src` is harmless -- its read is at `T+1` and pins `value` to
/// live memory).
#[test]
fn memmove_constraints_pin_the_memset_gap() {
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );

    // Honest: dst = src + 8, on a wide row, a narrow row and the terminal row.
    let honest = generate_memmove_trace(&[
        set_row(16, true, false, 0x1000, 0x1008),
        set_row(8, false, false, 0x1008, 0x1010),
        set_row(0, false, true, 0x1010, 0x1018),
    ]);
    assert!(
        validate_busless(&air, &honest),
        "a memset chain with dst = src + 8 must be accepted"
    );

    // The forgery: dst == src leaves `value` unconstrained.
    let degenerate = generate_memmove_trace(&[
        set_row(16, true, false, 0x1000, 0x1000),
        set_row(0, false, true, 0x1000, 0x1000),
    ]);
    assert!(
        !validate_busless(&air, &degenerate),
        "dst == src must be rejected: it makes every value lane a free field element"
    );

    // Wrong gap, and the wrong direction, are both out of contract too.
    for (src, dst, why) in [
        (0x1000u64, 0x1004u64, "a gap under one row width"),
        (0x1000, 0x1020, "a gap over one row width"),
        (
            0x1008,
            0x1000,
            "dst below src, which propagates the wrong way",
        ),
    ] {
        let trace = generate_memmove_trace(&[
            set_row(16, true, false, src, dst),
            set_row(0, false, true, src, dst),
        ]);
        assert!(
            !validate_busless(&air, &trace),
            "{why} must be rejected (src {src:#x}, dst {dst:#x})"
        );
    }

    // The high limb is pinned as well, so the gap cannot be forged across limbs.
    let straddle = generate_memmove_trace(&[
        set_row(16, true, false, 0x1000, 0x1_0000_1008),
        set_row(0, false, true, 0x1000, 0x1_0000_1008),
    ]);
    assert!(
        !validate_busless(&air, &straddle),
        "a gap of 8 in the low limb but not the high one must be rejected"
    );

    // And the gate really is `is_set`. This has to be a genuinely aliased copy —
    // `row()` hardcodes src 0x1000 / dst 0x2000, so using it here would only show
    // that the constraints tolerate a gap of 0x1000, not that they are off for Copy.
    // A memcpy with dst == src is harmless: its read is at T+1 and pins `value` to
    // live memory, which is exactly why the pin is gated on `is_set`.
    let copy = crate::tables::memmove::Functionality::Copy;
    let copy_aliased = generate_memmove_trace(&[
        row_of_width(copy, 8, 8, true, false, 0x1000, 0x1000),
        row_of_width(copy, 0, 1, false, true, 0x1008, 0x1008),
    ]);
    assert!(
        validate_busless(&air, &copy_aliased),
        "constraints 32-33 must not fire on Copy rows, even with dst == src"
    );
}

/// Constraint 14, `(1 - tail) * lt8 = 0`, in both directions.
///
/// This is the constraint that replaced the old DMA table's hard pin of
/// `tail = (count < 8)`. Erik asked for exactly this relaxation so the prover may
/// take one-byte rows at any count and reach the aligned `MEMW_A` path, so both
/// directions matter: the narrow-at-high-count row must be ACCEPTED (it is the
/// alignment prologue `memmove_row_width` emits), and the wide-at-low-count row must
/// be REJECTED (it would move eight bytes where fewer were authorised).
///
/// Neither case is expressible through `row()` or `set_row()`, which derive width
/// from count and so can only ever produce `tail == lt8`.
#[test]
fn memmove_constraint_14_frees_narrow_rows_but_not_wide_ones() {
    use crate::tables::memmove::Functionality::Copy;
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );

    // ACCEPTED: a one-byte row with eight bytes still to go — the prologue that
    // walks `dst` up to eight-byte alignment. `tail = 1`, `lt8 = 0`.
    let prologue = generate_memmove_trace(&[
        row_of_width(Copy, 16, 1, true, false, 0x1001, 0x2001),
        row_of_width(Copy, 15, 1, false, false, 0x1002, 0x2002),
        row_of_width(Copy, 14, 8, false, false, 0x1003, 0x2003),
        row_of_width(Copy, 6, 1, false, false, 0x100B, 0x200B),
        row_of_width(Copy, 0, 1, false, true, 0x100C, 0x200C),
    ]);
    assert!(
        validate_busless(&air, &prologue),
        "a one-byte row at count >= 8 is legal: it is the alignment prologue"
    );

    // REJECTED: an eight-byte row with fewer than eight bytes left. `tail = 0`,
    // `lt8 = 1`, so `(1 - tail) * lt8 = 1`.
    let overrun = generate_memmove_trace(&[
        row_of_width(Copy, 7, 8, true, false, 0x1000, 0x2000),
        row_of_width(Copy, 0, 1, false, true, 0x1008, 0x2008),
    ]);
    assert!(
        !validate_busless(&air, &overrun),
        "an eight-byte row must be illegal when only seven bytes remain"
    );
}

/// The `Commit` functionality at constraint level, which had no negative coverage
/// at all: memcpy rows have four forgery tests and memset five, commit none. The
/// `prove_elfs` forgery helper excludes it by construction (`is_copy = !IS_SET &&
/// !IS_COMMIT`), so the accepting direction is exercised end to end but nothing ever
/// tried to break the COMMIT-domain gating.
///
/// Covers constraints 12 (one-hot), 13 (no selector on a padding row) and 18
/// (`mu_com_wide = mu_com * (1 - tail)`), the last being the structural successor of
/// the deleted DMA_SET `FILL_WIDE`, which had four negative tests and lost them all.
#[test]
fn memmove_constraints_gate_the_commit_functionality() {
    use crate::tables::memmove::Functionality::{Commit, Copy};
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );

    // Baseline: an honest commit chain. `dst` is the global byte index, so it starts
    // at 0 and the gap pin must not fire here — commit is not `is_set`.
    let honest = generate_memmove_trace(&[
        row_of_width(Commit, 12, 8, true, false, 0x3000, 0),
        row_of_width(Commit, 4, 1, false, false, 0x3008, 8),
        row_of_width(Commit, 3, 1, false, false, 0x3009, 9),
        row_of_width(Commit, 2, 1, false, false, 0x300A, 10),
        row_of_width(Commit, 1, 1, false, false, 0x300B, 11),
        row_of_width(Commit, 0, 1, false, true, 0x300C, 12),
    ]);
    assert!(
        validate_busless(&air, &honest),
        "an honest commit chain must be accepted"
    );

    // Constraint 12: a row cannot claim two functionalities. Setting `is_set` on a
    // commit row would buy the inverted timestamp order on a chain the COMMIT chip
    // authorised.
    //
    // This case has to be built on a chain whose addresses already satisfy the memset
    // gap pin (constraints 32-33), or those reject it first and the assertion passes
    // for the wrong reason — verified by mutation: neutering 12 alone left an earlier
    // version of this test green.
    let gap_clean = generate_memmove_trace(&[
        row_of_width(Commit, 8, 8, true, false, 0x3000, 0x3008),
        row_of_width(Commit, 0, 1, false, true, 0x3008, 0x3010),
    ]);
    assert!(
        validate_busless(&air, &gap_clean),
        "the gap-clean commit baseline must itself be accepted"
    );
    let mut one_hot = gap_clean.clone();
    one_hot.main_table.set_fe(0, cols::IS_SET, FE::one());
    assert!(
        !validate_busless(&air, &one_hot),
        "is_set and is_commit must not both be set (constraint 12)"
    );

    // Constraint 18: widen a one-byte commit row. `mu_com_wide` is what stops it
    // broadcasting seven spurious `(index, 0)` pairs onto the COMMIT bus, which the
    // verifier rebuilds from `public_output` — so a forgery here corrupts the output
    // fingerprint rather than merely wasting a row.
    let mut trace = honest.clone();
    trace.main_table.set_fe(1, cols::MU_COM_WIDE, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a one-byte commit row must not claim the wide lanes (constraint 18)"
    );

    // Constraint 13: no selector on a padding row. The chain above is six rows, so
    // the trace pads to eight and row 7 is padding with mu = 0.
    let mut trace = honest.clone();
    assert_eq!(
        trace.main_table.get_row(7)[cols::MU],
        FE::zero(),
        "row 7 is expected to be padding"
    );
    trace.main_table.set_fe(7, cols::IS_COMMIT, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a padding row must not carry a functionality selector (constraint 13)"
    );

    // And the mirror of the memset gate: `mu_ram` is off for commit, so the RAM write
    // is suppressed. Flipping it on is a commit row that also writes to RAM.
    let mut trace = honest.clone();
    trace.main_table.set_fe(0, cols::MU_RAM, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a commit row must not also claim the RAM write (constraint 16)"
    );

    // Control: the same forgeries on a Copy chain are a different matter — this only
    // establishes that the honest Copy baseline is clean, so the failures above are
    // attributable to the commit gating rather than to the row shapes.
    let copy_ok = generate_memmove_trace(&[
        row_of_width(Copy, 8, 8, true, false, 0x1000, 0x2000),
        row_of_width(Copy, 0, 1, false, true, 0x1008, 0x2008),
    ]);
    assert!(
        validate_busless(&air, &copy_ok),
        "Copy baseline must be clean"
    );
}
