//! REGISTER table for register initialization and finalization.
//!
//! Similar to PAGE table but for registers (is_register=1).
//! Provides initial and final tokens for the Memory bus to balance
//! register read/write operations from MEMW.
//!
//! ## Token Model
//!
//! - **REG-C1**: Receives initial token `(1, address, ts=0, init)` - balances MEMW's send on first access
//! - **REG-C2**: Sends final token `(1, address, timestamp, fini)` - balances MEMW's receive on last access
//!
//! ## Columns
//!
//! | Column | Type | Description |
//! |--------|------|-------------|
//! | offset | RowIndex | Byte offset within register space |
//! | init | Byte | Initial value (0 for all registers at start) |
//! | fini | Byte | Final value after execution |
//! | timestamp | DWordWL | Final timestamp (0 if never accessed) |

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Constants
// =========================================================================

/// Number of general-purpose registers (x0-x31).
pub const NUM_REGISTERS: usize = 32;

/// Bytes per register access (registers are 64-bit = 8 bytes).
pub const BYTES_PER_REGISTER: usize = 8;

/// Total number of register byte addresses.
/// Each register uses 8 byte addresses in the Memory bus.
pub const NUM_REGISTER_ADDRESSES: usize = NUM_REGISTERS * BYTES_PER_REGISTER;

// =========================================================================
// Column indices for REGISTER table
// =========================================================================

pub mod cols {
    /// offset: Row index / byte address within register space
    pub const OFFSET: usize = 0;

    /// init: Initial byte value (0 for all registers)
    pub const INIT: usize = 1;

    /// fini: Final byte value after execution
    pub const FINI: usize = 2;

    /// timestamp[0]: Final timestamp low word (0 if never accessed)
    pub const TIMESTAMP_LO: usize = 3;

    /// timestamp[1]: Final timestamp high word
    pub const TIMESTAMP_HI: usize = 4;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 5;
}

// =========================================================================
// Types
// =========================================================================

/// Final state for a single register byte address.
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalRegisterByteState {
    /// Final timestamp (0 if never accessed)
    pub timestamp: u64,
    /// Final byte value
    pub value: u8,
}

/// Map from register byte address to final state.
pub type FinalRegisterStateMap = HashMap<u64, FinalRegisterByteState>;

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the REGISTER trace table.
///
/// Creates a table with NUM_REGISTER_ADDRESSES rows (32 regs × 8 bytes = 256).
/// Each row represents one byte address in register space.
///
/// ## Arguments
///
/// * `final_state` - Map from register byte address to final (timestamp, value)
///
/// ## Returns
///
/// The trace table for registers.
pub fn generate_register_trace(
    final_state: &FinalRegisterStateMap,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for offset in 0..NUM_REGISTER_ADDRESSES {
        let byte_addr = offset as u64;
        let base = offset * cols::NUM_COLUMNS;

        // Offset (row index = byte address in register space)
        data[base + cols::OFFSET] = FE::from(offset as u64);

        // Initial value: all registers start at 0
        let init_value = 0u8;
        data[base + cols::INIT] = FE::from(init_value as u64);

        // Final state: if accessed use final, otherwise use initial
        let (timestamp, fini_value) = if let Some(state) = final_state.get(&byte_addr) {
            (state.timestamp, state.value)
        } else {
            // Never accessed: timestamp=0, fini=init
            (0, init_value)
        };

        data[base + cols::FINI] = FE::from(fini_value as u64);
        data[base + cols::TIMESTAMP_LO] = FE::from(timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_HI] = FE::from(timestamp >> 32);
    }

    // Padding rows (if num_rows > NUM_REGISTER_ADDRESSES)
    // Already zero-initialized, which is correct (init=fini=0, ts=0)

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the REGISTER table.
///
/// ## Bus Interactions
///
/// - REG-C1: memory[1, address, 0, init] - receiver, multiplicity -1
/// - REG-C2: memory[1, address, timestamp, fini] - sender, multiplicity 1
///
/// Note: is_register=1 (constant) to distinguish from memory (is_register=0).
pub fn bus_interactions() -> Vec<BusInteraction> {
    // Address is just the offset (0..255 for 32 regs × 8 bytes)
    // Stored in low word, high word is 0
    let address_lo = BusValue::Packed {
        start_column: cols::OFFSET,
        packing: Packing::Direct,
    };
    let address_hi = BusValue::constant(0);

    vec![
        // REG-C1: memory[1, address, 0, init] - receive initial token
        BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::One,
            vec![
                // is_register = 1 (registers, not memory)
                BusValue::constant(1),
                // address_lo = offset
                address_lo.clone(),
                // address_hi = 0
                address_hi.clone(),
                // timestamp_lo = 0 (initial)
                BusValue::constant(0),
                // timestamp_hi = 0
                BusValue::constant(0),
                // value = init
                BusValue::Packed {
                    start_column: cols::INIT,
                    packing: Packing::Direct,
                },
            ],
        ),
        // REG-C2: memory[1, address, timestamp, fini] - send final token
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::One,
            vec![
                // is_register = 1
                BusValue::constant(1),
                // address_lo = offset
                address_lo,
                // address_hi = 0
                address_hi,
                // timestamp_lo (final)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_LO,
                    packing: Packing::Direct,
                },
                // timestamp_hi (final)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_HI,
                    packing: Packing::Direct,
                },
                // value = fini
                BusValue::Packed {
                    start_column: cols::FINI,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

// =========================================================================
// Helper functions
// =========================================================================

/// Compute the base address for a register index.
/// Per trace_builder.rs: reg_addr = 2 * reg_idx
pub fn register_base_address(reg_idx: u8) -> u64 {
    2 * reg_idx as u64
}

/// Compute the byte addresses used by a register.
/// Returns 8 addresses: base, base+1, ..., base+7
pub fn register_byte_addresses(reg_idx: u8) -> [u64; 8] {
    let base = register_base_address(reg_idx);
    [
        base,
        base + 1,
        base + 2,
        base + 3,
        base + 4,
        base + 5,
        base + 6,
        base + 7,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_base_address() {
        assert_eq!(register_base_address(0), 0);
        assert_eq!(register_base_address(1), 2);
        assert_eq!(register_base_address(2), 4);
        assert_eq!(register_base_address(31), 62);
    }

    #[test]
    fn test_generate_register_trace_empty() {
        let final_state = FinalRegisterStateMap::new();
        let trace = generate_register_trace(&final_state);

        // Should have power-of-2 rows >= 256
        assert!(trace.num_rows() >= NUM_REGISTER_ADDRESSES);
        assert!(trace.num_rows().is_power_of_two());

        // Check first row (address 0, never accessed)
        assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::INIT), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::FINI), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP_LO), FE::zero());
    }

    #[test]
    fn test_generate_register_trace_with_access() {
        let mut final_state = FinalRegisterStateMap::new();
        // Register x5 byte 0 was written with value 0x42 at timestamp 100
        let addr = register_base_address(5); // = 10
        final_state.insert(
            addr,
            FinalRegisterByteState {
                timestamp: 100,
                value: 0x42,
            },
        );

        let trace = generate_register_trace(&final_state);

        // Row 10 (address 10) should have the final state
        assert_eq!(*trace.main_table.get(10, cols::OFFSET), FE::from(10u64));
        assert_eq!(*trace.main_table.get(10, cols::INIT), FE::zero()); // init is always 0
        assert_eq!(*trace.main_table.get(10, cols::FINI), FE::from(0x42u64));
        assert_eq!(
            *trace.main_table.get(10, cols::TIMESTAMP_LO),
            FE::from(100u64)
        );
    }

    #[test]
    fn test_bus_interactions() {
        let interactions = bus_interactions();
        assert_eq!(interactions.len(), 2); // C1, C2
    }
}
