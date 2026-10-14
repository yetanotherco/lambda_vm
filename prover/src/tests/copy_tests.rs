use crate::tables::copy::{CopyOperation, Functionality, cols, generate_copy_trace};
use crate::tables::types::{FE, VmTable};
use crate::test_utils::{busless_air, validate_busless};

fn air() -> crate::test_utils::VmAir {
    busless_air(cols::NUM_COLUMNS, crate::tables::copy::CopyConstraints)
}

/// One row with an explicitly chosen width, so a test can state `single` and `count`
/// independently.
fn row_of_width(
    functionality: Functionality,
    count: u64,
    width: u8,
    first: bool,
    end: bool,
    src: u64,
    dst: u64,
) -> CopyOperation {
    CopyOperation {
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

/// A whole honest chain, on the schedule the trace builder takes: eight bytes while
/// eight remain, then one byte a row. `end` lands on the row that moves the last
/// bytes — there is no terminal row.
fn chain(functionality: Functionality, src: u64, dst: u64, count: u64) -> Vec<CopyOperation> {
    let mut rows = Vec::new();
    let mut offset = 0u64;
    let mut remaining = count;
    let mut first = true;
    while remaining != 0 {
        let width = if remaining < 8 { 1 } else { 8 };
        rows.push(row_of_width(
            functionality,
            remaining,
            width,
            first,
            false,
            src + offset,
            dst + offset,
        ));
        first = false;
        offset += u64::from(width);
        remaining -= u64::from(width);
    }
    match rows.last_mut() {
        Some(last) => last.end = true,
        None => rows.push(CopyOperation {
            width: 1,
            functionality,
            timestamp: 100,
            src,
            dst,
            count: 0,
            first: true,
            end: true,
            value: [0; 8],
        }),
    }
    rows
}

/// A memset chain. The contract the AIR pins is `dst = src + 8`, so build it that
/// way by default and let the tests below break it deliberately.
fn set_chain(src: u64, dst: u64, count: u64) -> Vec<CopyOperation> {
    chain(Functionality::Set, src, dst, count)
}

#[test]
fn copy_trace_uses_eight_byte_rows_then_a_byte_tail() {
    let trace = generate_copy_trace(&chain(Functionality::Copy, 0x1000, 0x2000, 10));

    let wide = trace.main_table.get_row(0);
    assert_eq!(wide[cols::SINGLE], FE::zero());
    assert_eq!(wide[cols::SRC_INCR_0], FE::from(0x1008u64));
    assert_eq!(wide[cols::COUNT_DECR_0], FE::from(2u64));
    assert_eq!(wide[cols::END], FE::zero());

    let tail = trace.main_table.get_row(1);
    assert_eq!(tail[cols::SINGLE], FE::one());
    assert_eq!(tail[cols::COUNT_0], FE::from(2u64));
    assert_eq!(tail[cols::COUNT_DECR_0], FE::one());
    assert_eq!(tail[cols::END], FE::zero());

    // The chain stops on the row that moves the last byte: `count == step == 1`.
    let last = trace.main_table.get_row(2);
    assert_eq!(last[cols::COUNT_0], FE::one());
    assert_eq!(last[cols::COUNT_DECR_0], FE::zero());
    assert_eq!(last[cols::END], FE::one());

    assert_eq!(trace.main_table.height, 4, "three rows pad to four");
    assert_eq!(trace.main_table.get_row(3)[cols::MU], FE::zero());
}

#[test]
fn an_empty_copy_call_is_a_single_first_and_end_row() {
    let rows = chain(Functionality::Copy, 0x1000, 0x2000, 0);
    assert_eq!(rows.len(), 1);
    let trace = generate_copy_trace(&rows);
    let first = trace.main_table.get_row(0);
    assert_eq!(first[cols::FIRST], FE::one());
    assert_eq!(first[cols::END], FE::one());
    assert_eq!(first[cols::MU], FE::one());
    // `count - step` borrows out, which is the AIR's proxy for this state.
    for k in 0..4 {
        assert_eq!(first[cols::COUNT_DECR_0 + k], FE::from(0xFFFFu64));
    }
    assert!(validate_busless(&air(), &trace));
}

#[test]
fn copy_constraints_accept_valid_rows_and_reject_nonzero_tail_lanes() {
    let mut trace = generate_copy_trace(&chain(Functionality::Copy, 0x1000, 0x2000, 2));
    assert!(validate_busless(&air(), &trace));

    trace.main_table.set(0, cols::VALUE[1], FE::one());
    assert!(
        !validate_busless(&air(), &trace),
        "a one-byte row must not smuggle additional copied lanes"
    );
}

/// `restrict_end`, which replaced the `ZERO` lookup over `count_decr`.
///
/// `end` must mean `count == step`, or the degenerate `(count, first) = (0, 1)`.
/// Setting it early and clearing it late are both rejected, and the zero-length
/// state is only available on a `first` row.
#[test]
fn copy_constraints_pin_end_to_the_last_chunk() {
    let air = air();

    // Early: claim `end` on a row that still has bytes left.
    let mut early = generate_copy_trace(&chain(Functionality::Copy, 0x1000, 0x2000, 16));
    assert!(validate_busless(&air, &early), "the baseline must be clean");
    early.main_table.set(0, cols::END, FE::one());
    assert!(
        !validate_busless(&air, &early),
        "`end` on a row with count != step must be rejected"
    );

    // Late: clear `end` on the row that moves the last bytes. Nothing busless
    // rejects that on its own — the row simply chains on — so the AIR instead
    // catches the successor, which has `count == 0` without `first`.
    let mut late = generate_copy_trace(&[
        row_of_width(Functionality::Copy, 8, 8, true, false, 0x1000, 0x2000),
        row_of_width(Functionality::Copy, 0, 8, false, false, 0x1008, 0x2008),
    ]);
    assert!(
        !validate_busless(&air, &late),
        "a `count == 0` row that is not `first` must be rejected"
    );
    // ...and marking that successor `end` does not save it either.
    late.main_table.set(1, cols::END, FE::one());
    assert!(
        !validate_busless(&air, &late),
        "the zero-count state is only legal on a `first` row"
    );
}

#[test]
fn copy_padding_row_cannot_claim_first_or_end() {
    // `(first + end) * (1 - mu) = 0` is the sole guard that a padding row (mu = 0)
    // cannot masquerade as the first or last row of a copy — bitness alone accepts
    // first = 1 or end = 1, so nothing else rejects it. A padding row claiming
    // `first` would forge an ECALL receive; claiming `end` would close a chain.
    let air = air();
    let base = generate_copy_trace(&chain(Functionality::Copy, 0x1000, 0x2000, 2));
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
        "a padding row (mu = 0) must not claim to close a copy"
    );
}

/// The memset operand contract, in the direction that matters.
///
/// `dst == src` is the degenerate case: read and write address the same cell at
/// adjacent timestamps, so the memory argument closes on `value == value` and every
/// value lane becomes a free field element. `copy:c:set_gap` is the only thing that
/// rejects it, so it is tested in both directions, and a `Copy` row is tested to
/// confirm the gate is `is_set` and does not leak onto the copy path (a memcpy with
/// `dst == src` is harmless -- its read pins `value` to live memory).
#[test]
fn copy_constraints_pin_the_memset_gap() {
    let air = air();

    let honest = generate_copy_trace(&set_chain(0x1000, 0x1008, 16));
    assert!(
        validate_busless(&air, &honest),
        "a memset chain with dst = src + 8 must be accepted"
    );

    // The forgery: dst == src leaves `value` unconstrained.
    let degenerate = generate_copy_trace(&set_chain(0x1000, 0x1000, 16));
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
        let trace = generate_copy_trace(&set_chain(src, dst, 16));
        assert!(
            !validate_busless(&air, &trace),
            "{why} must be rejected (src {src:#x}, dst {dst:#x})"
        );
    }

    // The high limb is pinned as well, so the gap cannot be forged across limbs.
    let straddle = generate_copy_trace(&set_chain(0x1000, 0x1_0000_1008, 16));
    assert!(
        !validate_busless(&air, &straddle),
        "a gap of 8 in the low limb but not the high one must be rejected"
    );

    // And the gate really is `is_set`. A memcpy with dst == src is harmless: its
    // read pins `value` to live memory, which is exactly why the pin is gated.
    let copy_aliased = generate_copy_trace(&chain(Functionality::Copy, 0x1000, 0x1000, 8));
    assert!(
        validate_busless(&air, &copy_aliased),
        "the memset gap pin must not fire on Copy rows, even with dst == src"
    );
}

/// A one-byte row is legal at any count.
///
/// The `count < 8` ALU lookup is gone: a wide row that overshoots can no longer
/// reach a legal `end`, so nothing has to check the width directly. What is left is
/// the freedom in the other direction, which the builder does not exercise and which
/// re-tightening the rule would silently remove.
#[test]
fn a_narrow_row_is_legal_at_any_count() {
    let prologue = generate_copy_trace(&[
        row_of_width(Functionality::Copy, 16, 1, true, false, 0x1001, 0x2001),
        row_of_width(Functionality::Copy, 15, 1, false, false, 0x1002, 0x2002),
        row_of_width(Functionality::Copy, 14, 8, false, false, 0x1003, 0x2003),
        row_of_width(Functionality::Copy, 6, 1, false, false, 0x100B, 0x200B),
        row_of_width(Functionality::Copy, 5, 1, false, false, 0x100C, 0x200C),
        row_of_width(Functionality::Copy, 4, 1, false, false, 0x100D, 0x200D),
        row_of_width(Functionality::Copy, 3, 1, false, false, 0x100E, 0x200E),
        row_of_width(Functionality::Copy, 2, 1, false, false, 0x100F, 0x200F),
        row_of_width(Functionality::Copy, 1, 1, false, true, 0x1010, 0x2010),
    ]);
    assert!(
        validate_busless(&air(), &prologue),
        "a one-byte row at count >= 8 must be legal, whether or not the builder emits one"
    );
}

/// `dst_res` and the register it is written back to.
///
/// One `REG` interaction serves two jobs: the `void*` the `mem*` calls return, and
/// the commit index a `write` advances. The column has to follow the selector, and
/// so does the register address.
#[test]
fn copy_constraints_pin_dst_res_to_the_functionality() {
    let air = air();

    // Non-write: `dst_res == dst`, unconditionally.
    let mut copy = generate_copy_trace(&chain(Functionality::Copy, 0x1000, 0x2000, 8));
    assert!(validate_busless(&air, &copy));
    copy.main_table.set(0, cols::DST_RES_0, FE::from(0x2001u64));
    assert!(
        !validate_busless(&air, &copy),
        "a memcpy row must return its destination unchanged"
    );

    // Write: `dst_res == dst + count`.
    let mut write = generate_copy_trace(&chain(Functionality::Write, 0x3000, 40, 8));
    assert!(validate_busless(&air, &write));
    assert_eq!(
        *write.main_table.get(0, cols::DST_RES_0),
        FE::from(48u64),
        "dst_res advances the commit index by count"
    );
    write.main_table.set(0, cols::DST_RES_0, FE::from(40u64));
    assert!(
        !validate_busless(&air, &write),
        "a write row must advance the commit index"
    );
}

/// The `write` functionality at constraint level: the one-hot rule, the
/// no-selector-on-padding rule, and the COMMIT-bus wide gate.
#[test]
fn copy_constraints_gate_the_write_functionality() {
    let air = air();

    // Baseline: an honest `write` chain. `dst` is the global byte index, so it
    // starts at 0 and the memset gap pin must not fire — a write is not `is_set`.
    let honest = generate_copy_trace(&chain(Functionality::Write, 0x3000, 0, 12));
    assert!(
        validate_busless(&air, &honest),
        "an honest write chain must be accepted"
    );

    // One-hot (`is_set * is_write = 0`): a row cannot claim two functionalities.
    // Setting `is_set` on a write row would buy the inverted timestamp order on a
    // chain the `write` ecall authorised.
    //
    // This case has to be built on a chain whose addresses already satisfy the
    // memset gap pin, or that rejects it first and the assertion passes for the
    // wrong reason.
    let gap_clean = generate_copy_trace(&[row_of_width(
        Functionality::Write,
        8,
        8,
        true,
        true,
        0x3000,
        0x3008,
    )]);
    assert!(
        validate_busless(&air, &gap_clean),
        "the gap-clean write baseline must itself be accepted"
    );
    let mut one_hot = gap_clean.clone();
    one_hot.main_table.set_fe(0, cols::IS_SET, FE::one());
    assert!(
        !validate_busless(&air, &one_hot),
        "is_set and is_write must not both be set (the one-hot constraint)"
    );

    // `mu_com_wide = mu_com * (1 - single)`: widen a one-byte write row. It is what
    // stops it broadcasting seven spurious `(index, 0)` pairs onto the COMMIT bus,
    // which the verifier rebuilds from `public_output` — so a forgery here corrupts
    // the output fingerprint rather than merely wasting a row.
    let mut trace = honest.clone();
    assert_eq!(*trace.main_table.get(1, cols::SINGLE), FE::one());
    trace.main_table.set_fe(1, cols::MU_COM_WIDE, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a one-byte write row must not claim the wide lanes (mu_com_wide)"
    );

    // `(is_set + is_write) * (1 - mu) = 0`: no selector on a padding row.
    let mut trace = honest.clone();
    let padding = trace.main_table.height - 1;
    assert_eq!(
        trace.main_table.get_row(padding)[cols::MU],
        FE::zero(),
        "the last row is expected to be padding"
    );
    trace.main_table.set_fe(padding, cols::IS_WRITE, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "a padding row must not carry a functionality selector"
    );

    // `first_write = first * is_write`.
    let mut trace = honest;
    trace.main_table.set_fe(1, cols::FIRST_WRITE, FE::one());
    assert!(
        !validate_busless(&air, &trace),
        "first_write must follow first * is_write"
    );
}

/// The ECALL number is decoded out of the two selectors rather than received on
/// three separate interactions, so the arithmetic has to produce exactly `64`,
/// `-30` and `-31` and nothing else.
#[test]
fn the_ecall_tuple_decodes_the_three_syscall_numbers() {
    use crate::tables::copy::bus_interactions;
    use crate::tables::types::BusId;
    use executor::vm::instruction::execution::{
        DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_SYSCALL_NUMBER,
    };
    use stark::lookup::{BusValue, LinearTerm};

    let ecall = bus_interactions()
        .into_iter()
        .find(|i| i.bus_id == BusId::Ecall as u64)
        .expect("the ECALL receive");
    assert!(!ecall.is_sender, "COPY receives the ECALL");

    let eval = |value: &BusValue, is_set: u64, is_write: u64| -> FE {
        match value {
            BusValue::Linear(terms) => terms.iter().fold(FE::zero(), |acc, t| match t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    let v = if *column == cols::IS_SET {
                        is_set
                    } else if *column == cols::IS_WRITE {
                        is_write
                    } else {
                        panic!("unexpected column {column} in the ECALL tuple")
                    };
                    let c = FE::from(coefficient.unsigned_abs());
                    if *coefficient < 0 {
                        acc - c * FE::from(v)
                    } else {
                        acc + c * FE::from(v)
                    }
                }
                LinearTerm::ColumnUnsigned { .. } => panic!("unexpected unsigned column"),
                LinearTerm::Constant(c) => {
                    if *c < 0 {
                        acc - FE::from(c.unsigned_abs())
                    } else {
                        acc + FE::from(c.unsigned_abs())
                    }
                }
            }),
            other => panic!("expected a written-out linear form, got {other:?}"),
        }
    };

    for (is_set, is_write, number) in [
        (0, 0, DMA_MEMCPY_SYSCALL_NUMBER),
        (1, 0, DMA_MEMSET_SYSCALL_NUMBER),
        (0, 1, 64),
    ] {
        assert_eq!(
            eval(&ecall.values[2], is_set, is_write),
            FE::from(number & 0xFFFF_FFFF),
            "low limb for (is_set {is_set}, is_write {is_write})"
        );
        assert_eq!(
            eval(&ecall.values[3], is_set, is_write),
            FE::from(number >> 32),
            "high limb for (is_set {is_set}, is_write {is_write})"
        );
    }
}

/// `copy:c:read_dst` addresses `x10 + 244·is_write`, so one interaction covers both
/// the `mem*` destination register and the commit index in `x254`.
#[test]
fn the_destination_register_follows_is_write() {
    use crate::tables::copy::bus_interactions;
    use crate::tables::types::BusId;
    use stark::lookup::{BusValue, LinearTerm, Multiplicity};

    let reads: Vec<_> = bus_interactions()
        .into_iter()
        .filter(|i| {
            i.bus_id == BusId::Memw as u64
                && matches!(i.multiplicity, Multiplicity::Column(c) if c == cols::FIRST)
        })
        .collect();
    assert_eq!(reads.len(), 3, "x11, x12 and the selected destination");

    // The address sits at index 9: eight `old` lanes then `is_register`.
    let selected = reads
        .iter()
        .find(|i| matches!(&i.values[9], BusValue::Linear(terms) if terms.len() == 2))
        .expect("one of the three reads addresses a selected register");
    match &selected.values[9] {
        BusValue::Linear(terms) => assert!(
            matches!(
                terms.as_slice(),
                [
                    LinearTerm::Constant(20),
                    LinearTerm::Column {
                        coefficient: 488,
                        column
                    }
                ] if *column == cols::IS_WRITE
            ),
            "the address must be 2·10 + 2·244·is_write, got {terms:?}"
        ),
        other => panic!("expected a linear address, got {other:?}"),
    }
}

/// Pins the shape of the COMMIT-domain sends: eight `(index, value)` pairs, one per
/// byte, indexed off `dst` — not one eight-lane tuple per row.
///
/// The verifier rebuilds this bus from `public_output` alone and never learns where
/// one `write` ECALL ended and the next began. Per-byte pairs are what make its
/// rebuild independent of the prover's row schedule, which restarts at every ECALL.
#[test]
fn copy_commit_sends_one_pair_per_byte() {
    use crate::tables::copy::bus_interactions;
    use crate::tables::types::BusId;
    use stark::lookup::{BusValue, LinearTerm, Multiplicity, Packing};

    let commit_sends: Vec<_> = bus_interactions()
        .into_iter()
        .filter(|interaction| interaction.bus_id == BusId::Commit as u64)
        .collect();

    assert_eq!(commit_sends.len(), 8, "one COMMIT send per byte lane");

    for (lane, interaction) in commit_sends.iter().enumerate() {
        assert!(interaction.is_sender, "lane {lane} must send");

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

/// The chain bus carries `position + step` forward and `position` back.
///
/// If the two sides ever disagree nothing errors — the bus simply stops balancing,
/// on some fixture, far from here. The send reads the `_incr`/`_decr` quadruples and
/// the receive reads the plain `DWordWL` columns.
#[test]
fn the_chain_bus_sends_the_incremented_position() {
    use crate::tables::copy::bus_interactions;
    use crate::tables::types::BusId;
    use stark::lookup::{BusValue, Multiplicity, Packing};

    let interactions = bus_interactions();
    let send = interactions
        .iter()
        .find(|i| i.bus_id == BusId::CopyNext as u64 && i.is_sender)
        .expect("the forward chain sender");
    assert!(
        matches!(send.multiplicity, Multiplicity::Diff(a, b) if a == cols::MU && b == cols::END)
    );
    for (slot, start) in [
        (2, cols::SRC_INCR_0),
        (3, cols::DST_INCR_0),
        (4, cols::COUNT_DECR_0),
    ] {
        assert!(
            matches!(send.values[slot], BusValue::Packed { start_column, packing: Packing::DWordHL }
                if start_column == start),
            "chain slot {slot} must send the incremented quadruple at {start}"
        );
    }

    let receive = interactions
        .iter()
        .find(|i| i.bus_id == BusId::CopyNext as u64 && !i.is_sender)
        .expect("the backward chain receiver");
    assert!(
        matches!(receive.multiplicity, Multiplicity::Diff(a, b) if a == cols::MU && b == cols::FIRST)
    );
    for (slot, start) in [(2, cols::SRC_0), (3, cols::DST_0), (4, cols::COUNT_0)] {
        assert!(
            matches!(receive.values[slot], BusValue::Packed { start_column, packing: Packing::DWordWL }
                if start_column == start),
            "chain slot {slot} must receive the plain position at {start}"
        );
    }
}

#[test]
fn copy_bus_interactions_count() {
    use crate::tables::copy::bus_interactions;
    // 1 ECALL receive, 4 register accesses, 2 chain tuples, 16 IS_HALF range checks
    // (four each for `src_incr`, `dst_incr`, `count_decr` and `dst_res`), the two
    // memory accesses, and eight COMMIT-domain `(index, value)` pairs.
    assert_eq!(bus_interactions().len(), 1 + 4 + 2 + 16 + 2 + 8);
}

#[test]
fn copy_constraints_count_and_indices() {
    use crate::tables::copy::CopyConstraints;
    use stark::constraints::builder::ConstraintSet;
    let meta = CopyConstraints.meta();
    assert_eq!(meta.len(), 37);
    // Dense, idx-ordered.
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    // `restrict_end` and the two conditional ADDs are degree 3; nothing exceeds it.
    assert_eq!(CopyConstraints.max_degree(), 3);
}
