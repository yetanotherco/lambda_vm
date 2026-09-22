//! IS_B48 range-check chip tests.
//!
//! The chip has no polynomial constraints, so everything it claims rests on its
//! four bus interactions. These tests cover the trace it builds, the shape of
//! those interactions, and — the part that actually matters — that the three
//! `IS_HALF` sends are load-bearing: a limb outside `[0, 2^16)` is rejected even
//! when the `(Word, Half)` tuple it offers callers is left intact.

use stark::lookup::{BusValue, LinearTerm, Multiplicity, Packing};

use crate::tables::is_b48::{
    B48_LIMIT, B48Operation, bus_interactions, cols, generate_is_b48_trace, merge_operations,
};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{BusId, FE};
use crate::test_utils::run_asm_elf;
use crate::tests::prove_elfs_tests::{prove_and_verify_vm_minimal, weigh_the_bus};

fn ops(values: &[u64]) -> Vec<B48Operation> {
    values.iter().map(|&v| B48Operation::new(v)).collect()
}

// =============================================================================
// Operation / trace
// =============================================================================

#[test]
fn from_address_keeps_the_top_48_bits() {
    let op = B48Operation::from_address(0xDEAD_BEEF_1234_5678);
    assert_eq!(op.value, 0xDEAD_BEEF_1234);
    // Every address inside one 64 KiB page maps to the same request — that is
    // the sharing the chip exists for.
    assert_eq!(
        B48Operation::from_address(0x1_0000),
        B48Operation::from_address(0x1_FFFF)
    );
    assert_ne!(
        B48Operation::from_address(0x1_FFFF),
        B48Operation::from_address(0x2_0000)
    );
}

#[test]
fn limbs_decompose_the_value() {
    let op = B48Operation::new(0xAAAA_BBBB_CCCC);
    assert_eq!(op.limbs(), [0xCCCC, 0xBBBB, 0xAAAA]);
}

/// The top of the range is representable; one past it is not. A caller that let
/// a larger value through would write a non-`Half` limb into the trace, which
/// surfaces only as an unbalanced `IS_HALF` bus much later.
#[test]
fn the_largest_representable_value_is_accepted() {
    let op = B48Operation::new(B48_LIMIT - 1);
    assert_eq!(op.limbs(), [0xFFFF, 0xFFFF, 0xFFFF]);
}

#[test]
#[should_panic(expected = "does not fit in 48 bits")]
fn a_value_wider_than_48_bits_is_refused() {
    B48Operation::new(B48_LIMIT);
}

/// Repeats collapse onto one row carrying the request count. This is the whole
/// point of the chip: 25 keccak lanes sharing a prefix pay for three `IS_HALF`
/// lookups once, not 25 times.
#[test]
fn merge_dedups_and_sums_multiplicity() {
    let merged = merge_operations(&ops(&[7, 7, 9, 7]));
    assert_eq!(
        merged,
        vec![(B48Operation::new(7), 3), (B48Operation::new(9), 1)]
    );
}

/// Row order is a deterministic function of the values, not of the request
/// order: the trace commitment covers it, so two runs of the same program have
/// to produce the same table.
#[test]
fn row_order_does_not_depend_on_request_order() {
    let forwards = generate_is_b48_trace(&ops(&[5, 1, 9, 1]));
    let backwards = generate_is_b48_trace(&ops(&[1, 9, 1, 5]));
    assert_eq!(
        forwards.main_table.row_major_data(),
        backwards.main_table.row_major_data()
    );
}

#[test]
fn trace_holds_one_row_per_distinct_value_with_its_multiplicity() {
    let trace = generate_is_b48_trace(&ops(&[0x1_0000_0001, 0x1_0000_0001, 0xABCD]));
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
    // Two real rows, padded to the 4-row minimum.
    assert_eq!(trace.num_rows(), 4);

    let row_of = |row: usize| {
        (
            [
                *trace.main_table.get(row, cols::VALUE_0),
                *trace.main_table.get(row, cols::VALUE_1),
                *trace.main_table.get(row, cols::VALUE_2),
            ],
            *trace.main_table.get(row, cols::MU),
        )
    };

    // Sorted by value: 0xABCD before 0x1_0000_0001.
    assert_eq!(
        row_of(0),
        ([FE::from(0xABCDu64), FE::zero(), FE::zero()], FE::one())
    );
    assert_eq!(
        row_of(1),
        ([FE::from(1u64), FE::zero(), FE::from(1u64)], FE::from(2u64))
    );
}

/// Padding rows are all-zero, so `mu = 0` gates both the `IS_HALF` sends and the
/// `IS_B48` receive. A padding row that contributed would make "present but
/// empty" stop matching "absent" on the bus.
#[test]
fn padding_rows_are_inert() {
    let trace = generate_is_b48_trace(&ops(&[42]));
    for row in 1..trace.num_rows() {
        for col in 0..cols::NUM_COLUMNS {
            assert_eq!(
                *trace.main_table.get(row, col),
                FE::zero(),
                "padding row {row} column {col} must be zero"
            );
        }
    }
}

/// An empty op list still produces a well-formed table. Nothing builds one today
/// — `generate_optional` drops the chip instead — but the shape has to hold if
/// something ever does.
#[test]
fn no_requests_still_yields_a_padded_table() {
    let trace = generate_is_b48_trace(&[]);
    assert_eq!(trace.num_rows(), 4);
    assert!(
        trace
            .main_table
            .row_major_data()
            .iter()
            .all(|v| *v == FE::zero()),
        "an empty table must contribute nothing"
    );
}

// =============================================================================
// Bus interactions
// =============================================================================

#[test]
fn bus_interaction_shape() {
    let interactions = bus_interactions();
    assert_eq!(interactions.len(), 4, "3 IS_HALF sends + 1 IS_B48 receive");

    for (i, interaction) in interactions.iter().enumerate() {
        assert!(
            matches!(interaction.multiplicity, Multiplicity::Column(cols::MU)),
            "interaction {i} must be gated on mu, or padding rows contribute"
        );
    }

    // The three limb checks, in column order.
    for (i, col) in [cols::VALUE_0, cols::VALUE_1, cols::VALUE_2]
        .into_iter()
        .enumerate()
    {
        assert_eq!(interactions[i].bus_id, u64::from(BusId::IsHalfword));
        assert!(interactions[i].is_sender);
        assert!(matches!(
            interactions[i].values.as_slice(),
            [BusValue::Packed { start_column, packing: Packing::Direct }] if *start_column == col
        ));
    }

    // The lookup this chip provides: IS_B48[value[1] + 2^16*value[2], value[0]].
    let provide = &interactions[3];
    assert_eq!(provide.bus_id, u64::from(BusId::IsB48));
    assert!(!provide.is_sender, "the chip receives on its own bus");
    assert_eq!(provide.values.len(), 2, "signature is IS_B48[Word, Half]");
    match &provide.values[0] {
        BusValue::Linear(terms) => {
            let mut seen: Vec<(i64, usize)> = terms
                .iter()
                .map(|t| match t {
                    LinearTerm::Column {
                        coefficient,
                        column,
                    } => (*coefficient, *column),
                    other => panic!("unexpected term {other:?} in the Word limb"),
                })
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, vec![(1, cols::VALUE_1), (1 << 16, cols::VALUE_2)]);
        }
        other => panic!("the Word limb must be a linear combination, got {other:?}"),
    }
    assert!(matches!(
        provide.values[1],
        BusValue::Packed {
            start_column: cols::VALUE_0,
            packing: Packing::Direct
        }
    ));
}

// =============================================================================
// End to end, through the real trace builder and verifier
// =============================================================================

/// KECCAK is the only chip sending on this bus today. One call's 25 lane
/// pointers span 200 bytes, so unless the state straddles a 64 KiB boundary they
/// share one prefix — and the table the builder produces is that one row with
/// `mu = 25`.
#[test]
fn a_keccak_run_collapses_its_lane_pointers_onto_one_row() {
    let (elf, logs, _instructions) = run_asm_elf("test_keccak");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    assert_eq!(traces.is_b48s.len(), 1, "a keccak run carries the chip");
    let trace = &traces.is_b48s[0];

    let total_mu: u64 = (0..trace.num_rows())
        .map(|row| *trace.main_table.get(row, cols::MU).value())
        .sum();
    let real_rows = (0..trace.num_rows())
        .filter(|&row| *trace.main_table.get(row, cols::MU) != FE::zero())
        .count();

    let keccak_calls = {
        let keccak = &traces.keccaks[0];
        (0..keccak.num_rows())
            .filter(|&row| {
                *keccak.main_table.get(row, crate::tables::keccak::cols::MU) != FE::zero()
            })
            .count() as u64
    };
    assert_eq!(
        total_mu,
        25 * keccak_calls,
        "every lane pointer of every call asks exactly once"
    );
    assert!(
        real_rows <= 2,
        "a 200-byte state touches at most two 64 KiB prefixes, got {real_rows} rows"
    );
}

/// A run with no keccak sends nothing on the bus, so the chip drops out.
#[test]
fn a_run_without_keccak_omits_the_chip() {
    let (elf, logs, _instructions) = run_asm_elf("xori");
    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(traces.is_b48s.is_empty());
    assert_eq!(traces.table_counts().is_b48, 0);
}

#[test]
fn an_honest_keccak_run_proves_and_verifies() {
    let (elf, logs, _instructions) = run_asm_elf("test_keccak");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "the honest proof must verify, or the negative below proves nothing"
    );
}

/// The negative control for the whole chip.
///
/// The forgery keeps the `(Word, Half)` tuple the chip offers exactly as it was
/// — `value[1] += 2^16` and `value[2] -= 1` leave `value[1] + 2^16*value[2]`
/// unchanged — so KECCAK's `IS_B48` sends still match and that bus balances. All
/// that changed is that `value[1]` is no longer a `Half`. If the three `IS_HALF`
/// sends were decorative, this would verify; the point is that they are not.
#[test]
fn a_limb_outside_the_half_range_is_caught_even_when_the_tuple_is_intact() {
    let (elf, logs, _instructions) = run_asm_elf("test_keccak");
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    assert!(
        prove_and_verify_vm_minimal(&elf, &mut traces),
        "the honest proof must verify first"
    );

    let trace = &mut traces.is_b48s[0];
    let row = (0..trace.num_rows())
        .find(|&r| *trace.main_table.get(r, cols::MU) != FE::zero())
        .expect("the table has a real row");
    let limb_1 = *trace.main_table.get(row, cols::VALUE_1).value();
    let limb_2 = *trace.main_table.get(row, cols::VALUE_2).value();
    assert!(
        limb_2 >= 1,
        "this forgery needs a nonzero top limb to borrow from; got {limb_2}"
    );
    trace
        .main_table
        .set(row, cols::VALUE_1, FE::from(limb_1 + (1 << 16)));
    trace
        .main_table
        .set(row, cols::VALUE_2, FE::from(limb_2 - 1));

    let outcome = weigh_the_bus(&elf, &mut traces, true);
    assert!(!outcome.accepted, "a non-Half limb must not verify");
    assert!(
        outcome.accepted_with_target_moved,
        "the bus balance has to be the reason for the rejection, not some \
         other check falling over first"
    );
    assert_ne!(
        outcome.contribution_sum, outcome.target,
        "the forged limb must move the bus sum"
    );
}

/// Sanity check on the arithmetic the forgery above relies on: moving `2^16`
/// between the two limbs leaves the `Word` the bus carries untouched.
#[test]
fn the_forged_limbs_still_encode_the_same_word() {
    let honest = B48Operation::new(0x0002_0003_4567);
    let [_, h1, h2] = honest.limbs();
    let word = u64::from(h1) + (u64::from(h2) << 16);
    let forged_word = (u64::from(h1) + (1 << 16)) + ((u64::from(h2) - 1) << 16);
    assert_eq!(word, forged_word);
    assert!(
        u64::from(h1) + (1 << 16) > u64::from(u16::MAX),
        "and the forged low limb is no longer a Half"
    );
}
