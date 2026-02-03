//! MEMORY_INIT table for initial memory state.
//!
//! This table contains the initial memory state loaded from the ELF.
//! It is preprocessed and the verifier can compute the commitment directly from the ELF.
//!
//! ## Purpose
//!
//! Provides initial values for all memory locations at timestamp=0.
//! The Memory bus consumes these values when addresses are first accessed.
//!
//! ## Columns
//!
//! - `address`: DWordWL (2 cols) - byte address
//! - `value`: BaseField (1 col) - initial 32-bit value (word-aligned)
//! - `μ`: BaseField (1 col) - multiplicity
//!
//! ## Bus Interactions
//!
//! - **Sender**: Memory bus - sends initial values at t=0 for each address

use executor::elf::Elf;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for MEMORY_INIT table
// =========================================================================

/// Column definitions for the MEMORY_INIT table.
pub mod cols {
    /// address[0]: Address (low word, bits 0-31)
    pub const ADDRESS_0: usize = 0;
    /// address[1]: Address (high word, bits 32-63)
    pub const ADDRESS_1: usize = 1;

    /// value: Initial 32-bit value at this address
    pub const VALUE: usize = 2;

    /// μ: Multiplicity for bus interactions
    pub const MU: usize = 3;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 4;
}

/// Number of precomputed columns (ADDRESS_0, ADDRESS_1, VALUE).
/// The MU column varies per execution.
pub const NUM_PRECOMPUTED_COLS: usize = 3;

// =========================================================================
// Trace generation
// =========================================================================

use std::collections::HashMap;

/// Map from address to row index in the MEMORY_INIT trace table.
pub type AddrToRow = HashMap<u64, usize>;

/// Generates the MEMORY_INIT trace table directly from the ELF.
///
/// Creates one row per 32-bit word in the ELF's PT_LOAD segments.
/// All segments are included (code, data, BSS zeros).
///
/// Returns the trace table and a map from address to row index for use with
/// `update_multiplicities`. All multiplicities are initialized to 0.
///
/// ## Padding
///
/// Empty rows use address=0, value=0 which should never be accessed
/// (address 0 is reserved/invalid in typical ELF layouts).
pub fn generate_memory_init_trace(elf: &Elf) -> (TraceTable<GoldilocksField, GoldilocksExtension>, AddrToRow) {
    let mut entries = Vec::new();
    let mut addr_to_row = HashMap::new();

    // Iterate all segments (not just executable - includes data, BSS)
    for segment in &elf.data {
        for (i, &word) in segment.values.iter().enumerate() {
            let addr = segment.base_addr + (i as u64 * 4);
            addr_to_row.insert(addr, entries.len());
            entries.push((addr, word as u64));
        }
    }

    // Pad to next power of 2, minimum 2
    let num_entries = entries.len();
    let num_rows = num_entries.next_power_of_two().max(2);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    // Fill actual entries (MU = 0 initially)
    for (row_idx, (addr, value)) in entries.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Address as DWordWL
        data[base + cols::ADDRESS_0] = FE::from(addr & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_1] = FE::from(addr >> 32);

        // Value (32-bit word)
        data[base + cols::VALUE] = FE::from(*value);

        // MU = 0 (already zero from vec initialization)
    }

    // Padding rows: address=0, value=0, MU=0 (all zeros, already initialized)

    (TraceTable::new_main(data, cols::NUM_COLUMNS, 1), addr_to_row)
}

/// Updates multiplicities in the MEMORY_INIT trace table.
///
/// For each address in `lookups`, increments the MU column in the corresponding row.
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    addr_to_row: &AddrToRow,
    lookups: &[u64],
) {
    for &addr in lookups {
        if let Some(&row_idx) = addr_to_row.get(&addr) {
            let current = trace.main_table.get(row_idx, cols::MU);
            trace.main_table.set(row_idx, cols::MU, current + FE::one());
        }
    }
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the MEMORY_INIT table.
///
/// MEMORY_INIT is a **sender** to the Memory bus, providing initial values
/// at timestamp=0.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // MEMORY_INIT sends to Memory bus: (address, t=0, value)
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Column(cols::MU),
            vec![
                // address as DWordWL (2 bus elements)
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::DWordWL,
                },
                // timestamp = 0 (constant)
                BusValue::constant(0),
                // value as Direct (1 bus element)
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
/// (ADDRESS_0, ADDRESS_1, VALUE), matching how the prover commits to traces.
///
/// Used by both prover (sanity check) and verifier (soundness check).
pub fn compute_precomputed_commitment(elf: &Elf, options: &ProofOptions) -> Commitment {
    // Step 1: Generate trace (MU=0, we only need precomputed columns)
    let (trace, _addr_to_row) = generate_memory_init_trace(elf);
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

/// Compute MEMORY_INIT commitment directly from an ELF.
///
/// This is what the verifier uses - no executor needed.
pub fn commitment_from_elf(elf: &Elf, options: &ProofOptions) -> Commitment {
    compute_precomputed_commitment(elf, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_init_trace_empty_elf() {
        let elf = Elf {
            entry_point: 0x1000,
            data: vec![],
        };

        let (trace, addr_to_row) = generate_memory_init_trace(&elf);

        // Should have minimum 2 rows (power of 2 padding)
        assert_eq!(trace.num_rows(), 2);
        assert!(addr_to_row.is_empty());
    }

    #[test]
    fn test_memory_init_trace_single_segment() {
        use executor::elf::Segment;

        let elf = Elf {
            entry_point: 0x1000,
            data: vec![Segment {
                base_addr: 0x1000,
                values: vec![0x12345678, 0xDEADBEEF],
                is_executable: true,
            }],
        };

        let (trace, addr_to_row) = generate_memory_init_trace(&elf);

        // Should have 2 entries (power of 2)
        assert_eq!(trace.num_rows(), 2);
        assert_eq!(addr_to_row.len(), 2);

        // Check first entry
        assert_eq!(addr_to_row.get(&0x1000), Some(&0));
        assert_eq!(*trace.main_table.get(0, cols::ADDRESS_0), FE::from(0x1000u64));
        assert_eq!(*trace.main_table.get(0, cols::ADDRESS_1), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::VALUE), FE::from(0x12345678u64));

        // Check second entry
        assert_eq!(addr_to_row.get(&0x1004), Some(&1));
        assert_eq!(*trace.main_table.get(1, cols::ADDRESS_0), FE::from(0x1004u64));
        assert_eq!(*trace.main_table.get(1, cols::VALUE), FE::from(0xDEADBEEFu64));
    }

    #[test]
    fn test_update_multiplicities() {
        use executor::elf::Segment;

        let elf = Elf {
            entry_point: 0x1000,
            data: vec![Segment {
                base_addr: 0x1000,
                values: vec![0x12345678, 0xDEADBEEF, 0xCAFEBABE, 0x0],
                is_executable: true,
            }],
        };

        let (mut trace, addr_to_row) = generate_memory_init_trace(&elf);

        // Initially all MU = 0
        assert_eq!(*trace.main_table.get(0, cols::MU), FE::zero());

        // Update multiplicities
        let lookups = vec![0x1000, 0x1000, 0x1004]; // 0x1000 accessed twice
        update_multiplicities(&mut trace, &addr_to_row, &lookups);

        // Check multiplicities
        assert_eq!(*trace.main_table.get(0, cols::MU), FE::from(2u64)); // 0x1000
        assert_eq!(*trace.main_table.get(1, cols::MU), FE::from(1u64)); // 0x1004
        assert_eq!(*trace.main_table.get(2, cols::MU), FE::zero()); // 0x1008 not accessed
    }
}
