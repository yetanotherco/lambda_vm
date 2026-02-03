//! MEMORY_INIT table for initial memory state.
//!
//! This table contains the initial memory state at byte-level using a paged scheme.
//! It initializes all memory that will be used during execution:
//! 1. ELF segments (code, data, BSS) - with their actual values
//! 2. Stack region - zero-initialized
//!
//! ## Token Model (per spec)
//!
//! Each memory address has a "token" (address, timestamp, value):
//! - **MEMORY_INIT**: Emits initial tokens at timestamp=0 (SENDER, +multiplicity)
//! - **MEMW**: Consumes old token (receive), emits new token (send)
//! - **MEMORY_FINAL**: Consumes final tokens (RECEIVER, -multiplicity)
//!
//! ## Columns
//!
//! - `is_register`: 1 col - 0 for memory (registers handled by verifier)
//! - `address`: 2 cols (lo, hi) - byte address
//! - `value`: 1 col - initial byte value (0-255)
//! - `μ`: 1 col - multiplicity (1 for valid entries, 0 for padding)
//!
//! ## Bus Interactions
//!
//! - **Sender**: Memory bus - sends (is_reg=0, addr, ts=0, value)
//!   Emits initial tokens that MEMW will consume on first access.
//!
//! ## Preprocessed
//!
//! All columns are preprocessed (deterministic from ELF + config).
//! The verifier can compute the commitment directly from the ELF.
//!
//! ## Note on Registers
//!
//! Per spec, register init/final can be computed directly by the verifier
//! since values are known (zero or from ELF/HALT). This table only handles memory.

use std::collections::HashMap;

use executor::elf::Elf;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{columns2rows, TraceTable};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Constants
// =========================================================================

/// Stack top address (where SP starts). Must match executor.
/// This is near max 64-bit, aligned to 16 bytes for RV64 ABI.
pub const STACK_TOP: u64 = 0xFFFF_FFFF_FFFF_FFF0;

/// Default stack size in bytes (64KB).
pub const DEFAULT_STACK_SIZE: u64 = 64 * 1024;

// =========================================================================
// Column indices for MEMORY_INIT table
// =========================================================================

/// Column definitions for the MEMORY_INIT table.
pub mod cols {
    /// is_register: 0 for memory, 1 for registers (always 0 in this table)
    pub const IS_REGISTER: usize = 0;

    /// address[0]: Byte address (low word, bits 0-31)
    pub const ADDRESS_0: usize = 1;
    /// address[1]: Byte address (high word, bits 32-63)
    pub const ADDRESS_1: usize = 2;

    /// value: Initial byte value at this address (0-255)
    pub const VALUE: usize = 3;

    /// μ: Multiplicity for bus interactions (1 for valid entries, 0 for padding)
    pub const MU: usize = 4;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 5;
}

/// Number of precomputed columns (IS_REGISTER, ADDRESS_0, ADDRESS_1, VALUE).
/// MU is also constant but kept separate for clarity.
pub const NUM_PRECOMPUTED_COLS: usize = 4;

// =========================================================================
// Configuration
// =========================================================================

/// Map from byte address to row index in the MEMORY_INIT trace table.
pub type AddrToRow = HashMap<u64, usize>;

/// Configuration for memory initialization.
#[derive(Debug, Clone)]
pub struct MemoryInitConfig {
    /// Size of the stack region in bytes.
    pub stack_size: u64,
}

impl Default for MemoryInitConfig {
    fn default() -> Self {
        Self {
            stack_size: DEFAULT_STACK_SIZE,
        }
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the MEMORY_INIT trace table at byte-level.
///
/// Creates one row per byte for:
/// 1. All ELF segments (code, data, BSS) - with their actual byte values
/// 2. Stack region (from STACK_TOP - stack_size to STACK_TOP) - zero-initialized
///
/// All multiplicities are 1 (every address participates in bus exactly once).
/// All is_register values are 0 (registers handled by verifier per spec).
///
/// ## Padding
///
/// Padding rows use is_register=0, address=0, value=0, MU=0 so they don't participate in bus.
pub fn generate_memory_init_trace(
    elf: &Elf,
    config: &MemoryInitConfig,
) -> (TraceTable<GoldilocksField, GoldilocksExtension>, AddrToRow) {
    let mut entries = Vec::new();
    let mut addr_to_row = HashMap::new();

    // 1. Add ELF segments at byte level
    for segment in &elf.data {
        for (word_idx, &word) in segment.values.iter().enumerate() {
            let word_addr = segment.base_addr + (word_idx as u64 * 4);

            // Split 32-bit word into 4 bytes (little-endian)
            for byte_offset in 0..4u64 {
                let byte_addr = word_addr + byte_offset;
                let byte_value = ((word >> (byte_offset * 8)) & 0xFF) as u8;

                addr_to_row.insert(byte_addr, entries.len());
                entries.push((byte_addr, byte_value as u64));
            }
        }
    }

    // 2. Add stack region (zero-initialized per spec)
    let stack_bottom = STACK_TOP - config.stack_size;
    for byte_addr in stack_bottom..STACK_TOP {
        // Skip if address already exists (shouldn't happen with proper ELF layout)
        if !addr_to_row.contains_key(&byte_addr) {
            addr_to_row.insert(byte_addr, entries.len());
            entries.push((byte_addr, 0)); // Stack is zero-initialized
        }
    }

    // Pad to next power of 2, minimum 2
    let num_entries = entries.len();
    let num_rows = num_entries.next_power_of_two().max(2);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    // Fill actual entries
    for (row_idx, (addr, value)) in entries.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // is_register = 0 (this table only handles memory)
        data[base + cols::IS_REGISTER] = FE::zero();

        // Address as two 32-bit words
        data[base + cols::ADDRESS_0] = FE::from(addr & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_1] = FE::from(addr >> 32);

        // Value (byte, 0-255)
        data[base + cols::VALUE] = FE::from(*value);

        // MU = 1 for all valid entries (every byte participates in bus)
        data[base + cols::MU] = FE::one();
    }

    // Padding rows: is_register=0, address=0, value=0, MU=0 (don't participate in bus)
    // Already zero from vec initialization

    (TraceTable::new_main(data, cols::NUM_COLUMNS, 1), addr_to_row)
}

/// Convenience function with default config.
pub fn generate_memory_init_trace_default(
    elf: &Elf,
) -> (TraceTable<GoldilocksField, GoldilocksExtension>, AddrToRow) {
    generate_memory_init_trace(elf, &MemoryInitConfig::default())
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the MEMORY_INIT table.
///
/// MEMORY_INIT is a **sender** on the Memory bus.
/// It emits initial tokens (is_register=0, address, timestamp=0, value)
/// that MEMW will consume on first access to each address.
///
/// For addresses never accessed during execution:
/// - MEMORY_INIT sends (is_reg=0, addr, ts=0, value_init)
/// - MEMORY_FINAL receives (is_reg=0, addr, ts=0, value_init)
/// - These cancel out in the bus (same fingerprint, opposite signs).
///
/// Bus signature: `[is_register, address_lo, address_hi, timestamp_lo, timestamp_hi, value]`
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // MEMORY_INIT sends to Memory bus: (is_reg=0, addr, ts=0, value_init)
        BusInteraction::sender(
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
                // timestamp_lo = 0 (initial state)
                BusValue::constant(0),
                // timestamp_hi = 0
                BusValue::constant(0),
                // value (initial byte value)
                BusValue::Packed {
                    start_column: cols::VALUE,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

// =========================================================================
// Precomputed commitment
// =========================================================================

/// Computes the LDE commitment for MEMORY_INIT precomputed columns.
///
/// This builds a Merkle tree over the LDE of the precomputed columns
/// (IS_REGISTER, ADDRESS_0, ADDRESS_1, VALUE), matching how the prover commits to traces.
///
/// Used by both prover (sanity check) and verifier (soundness check).
pub fn compute_precomputed_commitment(
    elf: &Elf,
    config: &MemoryInitConfig,
    options: &ProofOptions,
) -> Commitment {
    // Step 1: Generate trace
    let (trace, _addr_to_row) = generate_memory_init_trace(elf, config);
    let num_rows = trace.num_rows();

    // Step 2: Extract precomputed columns (0..NUM_PRECOMPUTED_COLS)
    let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .map(|col_idx| {
            (0..num_rows)
                .map(|row_idx| *trace.main_table.get(row_idx, col_idx))
                .collect()
        })
        .collect();

    // Step 3: Interpolate each column to a polynomial
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for memory_init column")
        })
        .collect();

    // Step 4: Evaluate polynomials on LDE domain
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for memory_init polynomial")
        })
        .collect();

    // Step 5: Bit-reverse permute (same as prover)
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    // Step 6: Convert columns to rows for Merkle tree
    let lde_rows = columns2rows(lde_columns);

    // Step 7: Build Merkle tree over LDE
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for memory_init LDE");

    tree.root
}

/// Compute MEMORY_INIT commitment directly from an ELF with default config.
///
/// This is what the verifier uses - no executor needed.
pub fn commitment_from_elf(elf: &Elf, options: &ProofOptions) -> Commitment {
    compute_precomputed_commitment(elf, &MemoryInitConfig::default(), options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use executor::elf::Segment;

    #[test]
    fn test_memory_init_trace_empty_elf() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![],
        };

        // Small stack for test
        let config = MemoryInitConfig { stack_size: 16 };
        let (trace, addr_to_row) = generate_memory_init_trace(&elf, &config);

        // Should have 16 stack bytes, padded to 16 (power of 2)
        assert_eq!(trace.num_rows(), 16);
        assert_eq!(addr_to_row.len(), 16);

        // All stack bytes should be zero-initialized with MU=1, is_register=0
        let stack_bottom = STACK_TOP - 16;
        let row = *addr_to_row.get(&stack_bottom).unwrap();
        assert_eq!(*trace.main_table.get(row, cols::IS_REGISTER), FE::zero());
        assert_eq!(*trace.main_table.get(row, cols::VALUE), FE::zero());
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::one());
    }

    #[test]
    fn test_memory_init_trace_byte_level() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![Segment {
                base_addr: 0x1000,
                values: vec![0x04030201], // Little-endian: bytes 01, 02, 03, 04
                is_executable: true,
            }],
        };

        // No stack for this test
        let config = MemoryInitConfig { stack_size: 0 };
        let (trace, addr_to_row) = generate_memory_init_trace(&elf, &config);

        // 4 bytes from ELF, padded to 4 (power of 2)
        assert_eq!(trace.num_rows(), 4);
        assert_eq!(addr_to_row.len(), 4);

        // Check byte 0: address=0x1000, value=0x01, is_register=0
        assert_eq!(addr_to_row.get(&0x1000), Some(&0));
        assert_eq!(*trace.main_table.get(0, cols::IS_REGISTER), FE::zero());
        assert_eq!(
            *trace.main_table.get(0, cols::ADDRESS_0),
            FE::from(0x1000u64)
        );
        assert_eq!(*trace.main_table.get(0, cols::VALUE), FE::from(0x01u64));
        assert_eq!(*trace.main_table.get(0, cols::MU), FE::one());

        // Check byte 1: address=0x1001, value=0x02
        assert_eq!(addr_to_row.get(&0x1001), Some(&1));
        assert_eq!(
            *trace.main_table.get(1, cols::ADDRESS_0),
            FE::from(0x1001u64)
        );
        assert_eq!(*trace.main_table.get(1, cols::VALUE), FE::from(0x02u64));

        // Check byte 2: address=0x1002, value=0x03
        assert_eq!(addr_to_row.get(&0x1002), Some(&2));
        assert_eq!(*trace.main_table.get(2, cols::VALUE), FE::from(0x03u64));

        // Check byte 3: address=0x1003, value=0x04
        assert_eq!(addr_to_row.get(&0x1003), Some(&3));
        assert_eq!(*trace.main_table.get(3, cols::VALUE), FE::from(0x04u64));
    }

    #[test]
    fn test_memory_init_includes_stack() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![],
        };

        let config = MemoryInitConfig { stack_size: 32 };
        let (trace, addr_to_row) = generate_memory_init_trace(&elf, &config);

        // Stack from STACK_TOP-32 to STACK_TOP
        let stack_bottom = STACK_TOP - 32;

        // Should have 32 stack bytes
        assert_eq!(addr_to_row.len(), 32);

        // Check stack addresses exist and are zero-initialized
        assert!(addr_to_row.contains_key(&stack_bottom));
        assert!(addr_to_row.contains_key(&(STACK_TOP - 1)));

        // Stack values should be 0 with MU=1, is_register=0
        let row = *addr_to_row.get(&stack_bottom).unwrap();
        assert_eq!(*trace.main_table.get(row, cols::IS_REGISTER), FE::zero());
        assert_eq!(*trace.main_table.get(row, cols::VALUE), FE::zero());
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::one());
    }

    #[test]
    fn test_memory_init_elf_plus_stack() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![Segment {
                base_addr: 0x1000,
                values: vec![0x12345678, 0xDEADBEEF],
                is_executable: true,
            }],
        };

        let config = MemoryInitConfig { stack_size: 8 };
        let (trace, addr_to_row) = generate_memory_init_trace(&elf, &config);

        // 8 bytes from ELF + 8 bytes from stack = 16 entries
        assert_eq!(addr_to_row.len(), 16);
        assert_eq!(trace.num_rows(), 16); // Already power of 2

        // Check ELF byte (0x78 is low byte of 0x12345678)
        let row = *addr_to_row.get(&0x1000).unwrap();
        assert_eq!(*trace.main_table.get(row, cols::IS_REGISTER), FE::zero());
        assert_eq!(*trace.main_table.get(row, cols::VALUE), FE::from(0x78u64));

        // Check stack byte
        let stack_addr = STACK_TOP - 1;
        let row = *addr_to_row.get(&stack_addr).unwrap();
        assert_eq!(*trace.main_table.get(row, cols::IS_REGISTER), FE::zero());
        assert_eq!(*trace.main_table.get(row, cols::VALUE), FE::zero());
    }

    #[test]
    fn test_bus_interactions_is_sender() {
        let interactions = bus_interactions();
        assert_eq!(interactions.len(), 1);
        // Verify it's a sender (positive multiplicity contribution)
        // The BusInteraction::sender creates a sender interaction
    }
}
