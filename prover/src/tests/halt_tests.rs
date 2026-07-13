//! Tests for the HALT (ECALL) table — single-`Word` timestamp and the `2^32-1`
//! register-finalization sentinel introduced by the timestamp narrowing.

use stark::lookup::{BusInteraction, BusValue, LinearTerm};

use crate::tables::halt::{bus_interactions, cols, generate_halt_trace};
use crate::tables::types::{BusId, FE};

/// Returns the constant of a pure-constant `BusValue` (a single `Constant` term),
/// or `None` if the value references any column.
fn const_value(v: &BusValue) -> Option<i64> {
    match v {
        BusValue::Linear(terms) if terms.len() == 1 => match terms[0] {
            LinearTerm::Constant(c) => Some(c),
            _ => None,
        },
        _ => None,
    }
}

#[test]
fn test_halt_trace_single_word_timestamp() {
    // Timestamp is a single `Word` column (not a 2-limb `DWordWL`): 3 columns total.
    assert_eq!(cols::NUM_COLUMNS, 3);

    let trace = generate_halt_trace(1234, 0x8000);
    assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
    assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP), FE::from(1234u64));
}

#[test]
fn test_halt_register_finalization_sentinel_and_widths() {
    let interactions = bus_interactions();

    // The register finalizations are the MEMW senders: x1-x9 + x11-x31 write 0 (30
    // tokens) and x10 is read (1 token) = 31.
    let memw_senders: Vec<&BusInteraction> = interactions
        .iter()
        .filter(|i| i.bus_id == BusId::Memw as u64 && i.is_sender)
        .collect();
    assert_eq!(
        memw_senders.len(),
        31,
        "31 register finalizations on the MEMW bus"
    );

    for tok in &memw_senders {
        match tok.values.len() {
            // Write token: [is_register, base[2], value[8], timestamp, w2, w4, w8].
            // A single-`Word` timestamp makes this 15 elements; ts is at index 11.
            15 => {
                assert_eq!(
                    const_value(&tok.values[11]),
                    Some(0xFFFF_FFFF),
                    "write finalization timestamp must be the 2^32-1 sentinel"
                );
            }
            // Read token (x10): old[8] then the 15-element write layout = 23 elements;
            // ts is at index 8 + 11 = 19.
            23 => {
                assert_eq!(
                    const_value(&tok.values[19]),
                    Some(0xFFFF_FFFF),
                    "x10 read finalization timestamp must be the 2^32-1 sentinel"
                );
                // old[0..8] = 0 enforces that x10 (the exit code) was 0 at halt time.
                for (i, v) in tok.values.iter().take(8).enumerate() {
                    assert_eq!(const_value(v), Some(0), "x10 old byte {i} must be 0");
                }
            }
            other => panic!("unexpected MEMW finalization token width {other} (expected 15 or 23)"),
        }
    }
}
