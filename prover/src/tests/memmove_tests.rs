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
        // A narrow row must leave lanes 1..7 clear (the `single * value[i] = 0` set), and the
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
/// makes `single == (count < 8)` identically, so the width rule cannot be stated.
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
    assert_eq!(wide[cols::SINGLE], FE::zero());
    assert_eq!(wide[cols::SRC_INCR_0], FE::from(0x1008u64));
    assert_eq!(wide[cols::COUNT_DECR_0], FE::from(2u64));
    for (i, &byte) in b"abcdefgh".iter().enumerate() {
        assert_eq!(wide[cols::VALUE[i]], FE::from(byte as u64));
    }

    let tail = trace.main_table.get_row(1);
    assert_eq!(tail[cols::SINGLE], FE::one());
    assert_eq!(tail[cols::SRC_INCR_0], FE::from(0x1001u64));
    assert_eq!(tail[cols::COUNT_DECR_0], FE::one());
    assert_eq!(tail[cols::VALUE[0]], FE::from(b'i' as u64));
    assert!(cols::VALUE[1..].iter().all(|&c| tail[c] == FE::zero()));

    let terminal = trace.main_table.get_row(3);
    assert_eq!(terminal[cols::END], FE::one());
    assert_eq!(terminal[cols::SINGLE], FE::one());
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

/// No-wraparound, which now lives in a range check rather than a constraint.
///
/// `src_incr`/`dst_incr` carry three halfwords; the top one is derived so that the
/// high-limb carry is zero by construction. That makes the old `carry_1 = 0`
/// constraint vacuous, and the property is carried instead by `IS_HALF` on `incr_2`
/// and on the derived halfword: both below `2^16` is exactly
/// `position_1 + carry_0 < 2^32`.
///
/// So a wrapping increment is no longer rejected by a polynomial and a busless test
/// cannot see it. What this does instead is evaluate the production `IS_HALF` value
/// -- the real `LinearTerm`s from `bus_interactions()`, not a copy of the formula --
/// against real rows, and check it lands in range for an honest increment and out of
/// range for a wrapping one.
#[test]
fn a_wrapping_increment_pushes_the_derived_halfword_out_of_range() {
    use crate::tables::memmove::{Functionality, bus_interactions};
    use crate::tables::types::{BusId, FE};
    use stark::lookup::{BusValue, LinearTerm};

    // The IS_HALF senders carrying a computed value are exactly the two derived
    // halfwords; the other range checks send plain columns.
    let derived: Vec<BusValue> = bus_interactions()
        .into_iter()
        .filter(|i| i.bus_id == BusId::IsHalfword as u64)
        .filter_map(|i| match i.values.into_iter().next() {
            Some(v @ BusValue::Linear(_)) => Some(v),
            _ => None,
        })
        .collect();
    assert_eq!(derived.len(), 2, "one derived halfword per position");

    let eval = |value: &BusValue, row: &[FE]| -> FE {
        match value {
            BusValue::Linear(terms) => terms.iter().fold(FE::zero(), |acc, t| match t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    let c = FE::from(coefficient.unsigned_abs());
                    if *coefficient < 0 {
                        acc - c * row[*column]
                    } else {
                        acc + c * row[*column]
                    }
                }
                LinearTerm::ColumnUnsigned {
                    coefficient,
                    column,
                } => acc + FE::from(*coefficient) * row[*column],
                LinearTerm::Constant(c) => acc + FE::from(c.unsigned_abs()),
            }),
            _ => unreachable!("filtered to Linear above"),
        }
    };

    let row_of = |src: u64, dst: u64| {
        generate_memmove_trace(&[MemmoveOperation {
            width: 8,
            functionality: Functionality::Copy,
            timestamp: 100,
            src,
            dst,
            count: 8,
            first: true,
            end: false,
            value: [0; 8],
        }])
    };

    // Honest: both derived halfwords are genuine halfwords.
    let honest = row_of(0x1234_5678_9ABC, 0x2000);
    let row: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *honest.main_table.get(0, c))
        .collect();
    for (i, value) in derived.iter().enumerate() {
        let v = eval(value, &row);
        assert!(
            (0..1u64 << 16).any(|h| FE::from(h) == v),
            "derived halfword {i} must be in [0, 2^16) on an honest row"
        );
    }

    // Wrapping source: `src + 8` rolls over 2^64, and the derived halfword lands at
    // 2^16 -- one past the top of the range, so IS_HALF has no such entry.
    let wrapped = row_of(u64::MAX - 3, 0x2000);
    let row: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *wrapped.main_table.get(0, c))
        .collect();
    assert_eq!(
        eval(&derived[0], &row),
        FE::from(1u64 << 16),
        "a wrapping source increment must push the derived halfword out of range"
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
    assert_eq!(meta.len(), 29);
    // Dense, idx-ordered.
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    // All constraints are degree 2 (no over-degree slips in a template change).
    assert_eq!(MemmoveConstraints.max_degree(), 2);
}

#[test]
fn memmove_padding_row_cannot_claim_first_or_end() {
    // `(first + end) * (1 - mu) = 0` is the sole guard that a
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
/// value lane becomes a free field element. The two `is_set` gap constraints are the only thing
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
        "the memset gap pin must not fire on Copy rows, even with dst == src"
    );
}

/// The width rule, now that it lives in a lookup rather than a constraint.
///
/// A one-byte row is legal at any count, and a wide row needs eight bytes remaining.
/// The second half used to be the polynomial `(1 - tail) * lt8 = 0`; it is now the
/// ALU lookup fired at multiplicity `mu - single` with its answer pinned to "not
/// less than eight". That is cheaper — it drops the `lt8` column — but it means the
/// rejecting direction is no longer visible to a busless test, so it is pinned here
/// by asserting the lookup's shape instead.
#[test]
fn a_narrow_row_is_legal_at_any_count() {
    use crate::tables::memmove::Functionality::Copy;
    let air = busless_air(
        cols::NUM_COLUMNS,
        crate::tables::memmove::MemmoveConstraints,
    );

    // The freedom the relaxation buys: one-byte rows with plenty of bytes left. The
    // builder does not emit these today, so nothing else in the tree covers it, and
    // re-tightening the rule would look like a harmless cleanup.
    let prologue = generate_memmove_trace(&[
        row_of_width(Copy, 16, 1, true, false, 0x1001, 0x2001),
        row_of_width(Copy, 15, 1, false, false, 0x1002, 0x2002),
        row_of_width(Copy, 14, 8, false, false, 0x1003, 0x2003),
        row_of_width(Copy, 6, 1, false, false, 0x100B, 0x200B),
        row_of_width(Copy, 0, 1, false, true, 0x100C, 0x200C),
    ]);
    assert!(
        validate_busless(&air, &prologue),
        "a one-byte row at count >= 8 must be legal, whether or not the builder emits one"
    );
}

/// The other half: a wide row must have eight bytes remaining.
///
/// This is an ALU lookup now, not a polynomial, so it cannot be exercised busless.
/// What is pinned here is that the lookup is wired the way the argument needs: fired
/// at `mu - single`, so it is skipped on narrow rows and can never go negative, and
/// asking for the answer "not less than eight" rather than storing it in a column.
#[test]
fn the_wide_row_check_is_a_lookup_fired_only_on_wide_rows() {
    use crate::tables::memmove::bus_interactions;
    use crate::tables::types::{BusId, alu_op};
    use stark::lookup::{BusValue, LinearTerm, Multiplicity, Packing};

    // A constant BusValue is a Linear with a single Constant term.
    let is_const = |v: &BusValue, want: i64| {
        matches!(v, BusValue::Linear(terms)
            if matches!(terms.as_slice(), [LinearTerm::Constant(c)] if *c == want))
    };

    let checks: Vec<_> = bus_interactions()
        .into_iter()
        .filter(|i| {
            i.bus_id == BusId::Alu as u64
                && matches!(i.multiplicity, Multiplicity::Diff(a, b)
                    if a == cols::MU && b == cols::SINGLE)
        })
        .collect();

    assert_eq!(
        checks.len(),
        1,
        "exactly one ALU lookup rides `mu - single`, so it is skipped on narrow rows \
         and its multiplicity can never go negative"
    );
    let c = &checks[0];

    assert!(
        matches!(c.values[0], BusValue::Packed { start_column, packing: Packing::DWordWL }
            if start_column == cols::COUNT_0),
        "it compares `count`"
    );
    assert!(is_const(&c.values[1], 8), "against 8");
    assert!(
        is_const(&c.values[3], alu_op::LT as i64),
        "with the LT opcode"
    );
    assert!(
        is_const(&c.values[4], 0),
        "and demands the answer be 0, i.e. NOT less than eight -- pinned as a \
         constant rather than stored in a column, which is where the saving is"
    );
}

/// The `Commit` functionality at constraint level, which had no negative coverage
/// at all: memcpy rows have four forgery tests and memset five, commit none. The
/// `prove_elfs` forgery helper excludes it by construction (`is_copy = !IS_SET &&
/// !IS_COMMIT`), so the accepting direction is exercised end to end but nothing ever
/// tried to break the COMMIT-domain gating.
///
/// Covers the one-hot `is_set * is_commit = 0`, the no-selector-on-padding rule, and
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

    // One-hot (`is_set * is_commit = 0`): a row cannot claim two functionalities. Setting `is_set` on a
    // commit row would buy the inverted timestamp order on a chain the COMMIT chip
    // authorised.
    //
    // This case has to be built on a chain whose addresses already satisfy the memset
    // memset gap pin, or those reject it first and the assertion passes
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
        "is_set and is_commit must not both be set (the one-hot constraint)"
    );

    // `mu_com_wide = mu_com * (1 - single)`: widen a one-byte commit row. It is what stops it
    // broadcasting seven spurious `(index, 0)` pairs onto the COMMIT bus, which the
    // verifier rebuilds from `public_output` — so a forgery here corrupts the output
    // fingerprint rather than merely wasting a row.
    let mut trace = honest.clone();
    trace.main_table.set_fe(1, cols::MU_COM_WIDE, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a one-byte commit row must not claim the wide lanes (mu_com_wide)"
    );

    // `(is_set + is_commit) * (1 - mu) = 0`: no selector on a padding row. The chain above is six rows, so
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
        "a padding row must not carry a functionality selector"
    );

    // There is deliberately no "commit row also claims the RAM write" case here any
    // more. That forgery needed `mu_ram` to be a witness column; the RAM write now
    // rides the linear multiplicity `mu - end - mu_com` directly, so a commit row
    // (`mu_com = mu - end`) drives it to zero by construction and there is nothing
    // left to forge.

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

/// The chain bus still carries `position + step`, in both halves.
///
/// This is the one that matters. `MEMMOVE_NEXT`'s send side used to be a
/// `Packing::DWordHL` over four columns; with the top halfword derived it is a
/// written-out linear form, and the receive side still reads a plain `DWordWL`. If the
/// two ever disagree nothing errors -- the bus simply stops balancing, on some fixture,
/// far from here. So evaluate the production `LinearTerm`s against real rows and check
/// they reproduce the low and high halves of `position + step` exactly.
#[test]
fn the_chain_bus_still_sends_position_plus_step() {
    use crate::tables::memmove::{Functionality, bus_interactions};
    use crate::tables::types::{BusId, FE};
    use stark::lookup::{BusValue, LinearTerm};

    let chain = bus_interactions()
        .into_iter()
        .find(|i| i.bus_id == BusId::MemmoveNext as u64 && i.is_sender)
        .expect("the forward chain sender");
    // [ts_lo, ts_hi, src_lo, src_hi, dst_lo, dst_hi, count_decr(2), is_set, is_commit]
    let (src_lo, src_hi) = (&chain.values[2], &chain.values[3]);
    let (dst_lo, dst_hi) = (&chain.values[4], &chain.values[5]);

    let eval = |value: &BusValue, row: &[FE]| -> FE {
        match value {
            BusValue::Linear(terms) => terms.iter().fold(FE::zero(), |acc, t| match t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    let c = FE::from(coefficient.unsigned_abs());
                    if *coefficient < 0 {
                        acc - c * row[*column]
                    } else {
                        acc + c * row[*column]
                    }
                }
                LinearTerm::ColumnUnsigned {
                    coefficient,
                    column,
                } => acc + FE::from(*coefficient) * row[*column],
                LinearTerm::Constant(c) => acc + FE::from(c.unsigned_abs()),
            }),
            other => panic!("expected a written-out linear form, got {other:?}"),
        }
    };

    // Alignments, widths and a high limb that is genuinely non-zero, plus a case whose
    // low limb carries into the high one.
    for (src, dst, width) in [
        (0x1000u64, 0x2000u64, 8u8),
        (0x1001, 0x2007, 8),
        (0x1000, 0x2000, 1),
        (0x1234_5678_9ABC, 0xFEDC_BA98_7654, 8),
        (0x0000_0000_FFFF_FFFC, 0x0000_0001_FFFF_FFFF, 8),
    ] {
        let trace = generate_memmove_trace(&[MemmoveOperation {
            width,
            functionality: Functionality::Copy,
            timestamp: 100,
            src,
            dst,
            count: 64,
            first: true,
            end: false,
            value: [0; 8],
        }]);
        let row: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(0, c))
            .collect();

        for (name, base, lo, hi) in [("src", src, src_lo, src_hi), ("dst", dst, dst_lo, dst_hi)] {
            let want = base.wrapping_add(width as u64);
            assert_eq!(
                eval(lo, &row),
                FE::from(want & 0xFFFF_FFFF),
                "{name} low half, base {base:#x} width {width}"
            );
            assert_eq!(
                eval(hi, &row),
                FE::from(want >> 32),
                "{name} high half, base {base:#x} width {width}"
            );
        }
    }
}
