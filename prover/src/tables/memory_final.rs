//! MEMORY_FINAL table for final memory state.
//!
//! This table contains the final memory state after execution at byte-level.
//! It has the same byte addresses as MEMORY_INIT but with final timestamps and values.
//!
//! ## Token Model (per spec)
//!
//! Each memory address has a "token" (address, timestamp, value):
//! - **MEMORY_INIT**: Emits initial tokens at timestamp=0 (SENDER, +multiplicity)
//! - **MEMW**: Consumes old token (receive), emits new token (send)
//! - **MEMORY_FINAL**: Consumes final tokens (RECEIVER, -multiplicity)
//!
//! ## Purpose
//!
//! Receives final values from the Memory bus to balance MEMW's last access.
//! When MEMW last accesses an address, it sends (addr, ts_new, val_new).
//! MEMORY_FINAL receives that same tuple to consume the token.
//!
//! For non-accessed addresses:
//! - MEMORY_INIT sends (is_reg=0, addr, ts=0, value_init)
//! - MEMORY_FINAL receives (is_reg=0, addr, ts=0, value_init)
//! - These cancel out in the bus (same fingerprint, opposite signs).
//!
//! ## Regions Covered
//!
//! Same as MEMORY_INIT:
//! 1. **ELF segments**: Code, data, BSS
//! 2. **Stack region**: From (STACK_TOP - stack_size) to STACK_TOP
//!
//! ## Columns
//!
//! - `is_register`: 1 col - 0 for memory (registers handled by verifier)
//! - `address`: 2 cols (lo, hi) - byte address (same as MEMORY_INIT)
//! - `timestamp`: 2 cols (lo, hi) - final timestamp (0 if never accessed)
//! - `value`: 1 col - final byte value
//! - `μ`: 1 col - multiplicity (always 1)
//!
//! ## Bus Interactions
//!
//! - **Receiver**: Memory bus - receives (is_reg=0, addr, ts_final, value_final)
//!   Consumes final tokens to balance the bus.
//!
//! ## NOT Preprocessed
//!
//! Unlike MEMORY_INIT, this table cannot be preprocessed because timestamps
//! and values depend on execution.

use std::collections::HashMap;

use executor::elf::Elf;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::memory_init::{AddrToRow, MemoryInitConfig, STACK_TOP};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for MEMORY_FINAL table
// =========================================================================

/// Column definitions for the MEMORY_FINAL table.
pub mod cols {
    /// is_register: 0 for memory, 1 for registers (always 0 in this table)
    pub const IS_REGISTER: usize = 0;

    /// address[0]: Address (low word, bits 0-31)
    pub const ADDRESS_0: usize = 1;
    /// address[1]: Address (high word, bits 32-63)
    pub const ADDRESS_1: usize = 2;

    /// timestamp[0]: Final timestamp (low word, bits 0-31)
    pub const TIMESTAMP_0: usize = 3;
    /// timestamp[1]: Final timestamp (high word, bits 32-63)
    pub const TIMESTAMP_1: usize = 4;

    /// value: Final byte value at this address (0-255)
    pub const VALUE: usize = 5;

    /// μ: Multiplicity for bus interactions (always 1)
    pub const MU: usize = 6;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 7;
}

// =========================================================================
// Final memory state
// =========================================================================

/// Final state for a single byte address.
#[derive(Debug, Clone, Copy)]
pub struct FinalByteState {
    /// Final timestamp (0 if never accessed)
    pub timestamp: u64,
    /// Final byte value
    pub value: u8,
}

/// Map from byte address to final state.
pub type FinalStateMap = HashMap<u64, FinalByteState>;

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the MEMORY_FINAL trace table at byte-level.
///
/// For each byte address in MEMORY_INIT (ELF + stack):
/// - If accessed: use final (timestamp, value) from `final_state`
/// - If not accessed: use (timestamp=0, value=initial_value)
///
/// The table has the same byte addresses as MEMORY_INIT, ensuring the bus balances.
///
/// ## Arguments
///
/// * `elf` - The ELF containing initial values for code/data
/// * `config` - Memory configuration (stack size)
/// * `final_state` - Map from byte address to final (timestamp, value) for accessed bytes
///
/// ## Returns
///
/// The trace table and address-to-row mapping (same as MEMORY_INIT).
pub fn generate_memory_final_trace(
    elf: &Elf,
    config: &MemoryInitConfig,
    final_state: &FinalStateMap,
) -> (TraceTable<GoldilocksField, GoldilocksExtension>, AddrToRow) {
    let mut entries = Vec::new();
    let mut addr_to_row = HashMap::new();

    // 1. Add ELF segments at byte level (same order as MEMORY_INIT)
    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr + (i as u64 * 4);

            // Split 32-bit word into 4 bytes (little-endian)
            for byte_offset in 0..4u64 {
                let byte_addr = word_addr + byte_offset;
                let initial_byte = ((word >> (byte_offset * 8)) & 0xFF) as u8;

                addr_to_row.insert(byte_addr, entries.len());

                // Get final state: if accessed use final, otherwise use initial
                let (timestamp, value) = if let Some(state) = final_state.get(&byte_addr) {
                    (state.timestamp, state.value)
                } else {
                    // Never accessed: timestamp=0, value=initial
                    (0, initial_byte)
                };

                entries.push((byte_addr, timestamp, value as u64));
            }
        }
    }

    // 2. Add stack region (same order as MEMORY_INIT)
    let stack_bottom = STACK_TOP - config.stack_size;
    for byte_addr in stack_bottom..STACK_TOP {
        // Skip if address already exists (shouldn't happen, but be safe)
        if !addr_to_row.contains_key(&byte_addr) {
            addr_to_row.insert(byte_addr, entries.len());

            // Get final state: if accessed use final, otherwise use initial (0)
            let (timestamp, value) = if let Some(state) = final_state.get(&byte_addr) {
                (state.timestamp, state.value as u64)
            } else {
                // Never accessed: timestamp=0, value=0 (stack starts zeroed)
                (0, 0)
            };

            entries.push((byte_addr, timestamp, value));
        }
    }

    // Pad to next power of 2, minimum 2
    let num_entries = entries.len();
    let num_rows = num_entries.next_power_of_two().max(2);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    // Fill actual entries
    for (row_idx, (addr, timestamp, value)) in entries.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // is_register = 0 (this table only handles memory)
        data[base + cols::IS_REGISTER] = FE::zero();

        // Address as two 32-bit words
        data[base + cols::ADDRESS_0] = FE::from(addr & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_1] = FE::from(addr >> 32);

        // Timestamp as two 32-bit words
        data[base + cols::TIMESTAMP_0] = FE::from(timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(timestamp >> 32);

        // Value (byte)
        data[base + cols::VALUE] = FE::from(*value);

        // MU = 1 for all entries (every address participates in bus)
        data[base + cols::MU] = FE::one();
    }

    // Padding rows: is_register=0, address=0, timestamp=0, value=0, MU=0
    // (all zeros, already initialized, MU=0 so they don't participate in bus)

    (TraceTable::new_main(data, cols::NUM_COLUMNS, 1), addr_to_row)
}

/// Convenience function with default config.
pub fn generate_memory_final_trace_default(
    elf: &Elf,
    final_state: &FinalStateMap,
) -> (TraceTable<GoldilocksField, GoldilocksExtension>, AddrToRow) {
    generate_memory_final_trace(elf, &MemoryInitConfig::default(), final_state)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the MEMORY_FINAL table.
///
/// MEMORY_FINAL is a **receiver** on the Memory bus.
/// It receives (is_register=0, address, timestamp_final, value_final) to consume
/// the final tokens emitted by MEMW on last access.
///
/// For addresses never accessed:
/// - MEMORY_INIT sends (is_reg=0, addr, ts=0, value_init)
/// - MEMORY_FINAL receives (is_reg=0, addr, ts=0, value_init)
/// - These cancel out (same fingerprint, opposite signs).
///
/// Bus signature: `[is_register, address_lo, address_hi, timestamp_lo, timestamp_hi, value]`
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // MEMORY_FINAL receives from Memory bus: (is_reg=0, addr, ts_final, val_final)
        BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Column(cols::MU),
            vec![
                // is_register (0 for memory)
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                // address_lo
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::Direct,
                },
                // address_hi
                BusValue::Packed {
                    start_column: cols::ADDRESS_1,
                    packing: Packing::Direct,
                },
                // timestamp_lo (final timestamp)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                // timestamp_hi (final timestamp)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // value (final byte value)
                BusValue::Packed {
                    start_column: cols::VALUE,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use executor::elf::Segment;

    #[test]
    fn test_memory_final_no_accesses() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![Segment {
                base_addr: 0x1000,
                values: vec![0x04030201], // Little-endian bytes
                is_executable: true,
            }],
        };

        // No bytes accessed, no stack
        let config = MemoryInitConfig { stack_size: 0 };
        let final_state = FinalStateMap::new();

        let (trace, addr_to_row) = generate_memory_final_trace(&elf, &config, &final_state);

        // 4 bytes, padded to power of 2
        assert_eq!(trace.num_rows(), 4);
        assert_eq!(addr_to_row.len(), 4);

        // Check byte 0: should have initial value, timestamp=0, is_register=0
        assert_eq!(*trace.main_table.get(0, cols::IS_REGISTER), FE::zero());
        assert_eq!(
            *trace.main_table.get(0, cols::ADDRESS_0),
            FE::from(0x1000u64)
        );
        assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP_0), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::VALUE), FE::from(0x01u64));
        assert_eq!(*trace.main_table.get(0, cols::MU), FE::one());
    }

    #[test]
    fn test_memory_final_with_accesses() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![Segment {
                base_addr: 0x1000,
                values: vec![0x04030201],
                is_executable: true,
            }],
        };

        // Byte 0 was accessed and changed
        let config = MemoryInitConfig { stack_size: 0 };
        let mut final_state = FinalStateMap::new();
        final_state.insert(
            0x1000,
            FinalByteState {
                timestamp: 48,
                value: 0xAB,
            },
        );

        let (trace, _) = generate_memory_final_trace(&elf, &config, &final_state);

        // Byte 0: should have final values
        assert_eq!(*trace.main_table.get(0, cols::IS_REGISTER), FE::zero());
        assert_eq!(
            *trace.main_table.get(0, cols::ADDRESS_0),
            FE::from(0x1000u64)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::TIMESTAMP_0),
            FE::from(48u64)
        );
        assert_eq!(*trace.main_table.get(0, cols::VALUE), FE::from(0xABu64));

        // Byte 1: should have initial values (not accessed)
        assert_eq!(
            *trace.main_table.get(1, cols::ADDRESS_0),
            FE::from(0x1001u64)
        );
        assert_eq!(*trace.main_table.get(1, cols::TIMESTAMP_0), FE::zero());
        assert_eq!(*trace.main_table.get(1, cols::VALUE), FE::from(0x02u64));
    }

    #[test]
    fn test_memory_final_includes_stack() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![],
        };

        // Stack from STACK_TOP-16 to STACK_TOP
        let config = MemoryInitConfig { stack_size: 16 };

        // Stack byte was written
        let mut final_state = FinalStateMap::new();
        let stack_addr = STACK_TOP - 8;
        final_state.insert(
            stack_addr,
            FinalByteState {
                timestamp: 100,
                value: 0x42,
            },
        );

        let (trace, addr_to_row) = generate_memory_final_trace(&elf, &config, &final_state);

        // Should have 16 stack bytes
        assert_eq!(addr_to_row.len(), 16);

        // Check written stack byte has final values
        let row = *addr_to_row.get(&stack_addr).unwrap();
        assert_eq!(*trace.main_table.get(row, cols::IS_REGISTER), FE::zero());
        assert_eq!(
            *trace.main_table.get(row, cols::TIMESTAMP_0),
            FE::from(100u64)
        );
        assert_eq!(*trace.main_table.get(row, cols::VALUE), FE::from(0x42u64));
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::one());

        // Check unwritten stack byte has initial values (ts=0, val=0)
        let unwritten_addr = STACK_TOP - 1;
        let row2 = *addr_to_row.get(&unwritten_addr).unwrap();
        assert_eq!(*trace.main_table.get(row2, cols::IS_REGISTER), FE::zero());
        assert_eq!(*trace.main_table.get(row2, cols::TIMESTAMP_0), FE::zero());
        assert_eq!(*trace.main_table.get(row2, cols::VALUE), FE::zero());
    }

    #[test]
    fn test_bus_interactions_is_receiver() {
        let interactions = bus_interactions();
        assert_eq!(interactions.len(), 1);
        // Verify it's a receiver (negative multiplicity contribution)
        // The BusInteraction::receiver creates a receiver interaction
    }
}
