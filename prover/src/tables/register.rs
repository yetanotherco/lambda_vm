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
//! | init | Word | Initial value (0 for all registers at start) |
//! | fini | Word | Final value after execution |
//! | timestamp | DWordWL | Final timestamp (0 if never accessed) |

use std::collections::HashMap;

use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use super::page::STACK_TOP;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Constants
// =========================================================================

/// Number of general-purpose registers (x0-x31).
pub const NUM_REGISTERS: usize = 32;

/// Words per register access (registers are 64-bit = 2 Words of 32 bits each).
/// Per spec: registers use write2=1 meaning 2 addresses accessed.
pub const WORDS_PER_REGISTER: usize = 2;

/// Total number of register Word addresses.
/// 32 registers (x0-x31) × 2 Words + 1 register (x255/PC) × 2 Words = 66.
pub const NUM_REGISTER_ADDRESSES: usize = NUM_REGISTERS * WORDS_PER_REGISTER + 2;

/// Number of preprocessed columns (OFFSET, INIT).
/// OFFSET contains the register addresses, INIT contains initial values.
pub const NUM_PREPROCESSED_COLS: usize = 2;

/// Word address of x255 (PC) low word: 2 * 255 = 510.
const PC_ADDR_LO: u64 = 510;
/// Word address of x255 (PC) high word: 511.
const PC_ADDR_HI: u64 = 511;

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

/// Final state for a single register Word address.
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalRegisterWordState {
    /// Final timestamp (0 if never accessed)
    pub timestamp: u64,
    /// Final Word value (32-bit)
    pub value: u32,
}

/// Map from register Word address to final state.
pub type FinalRegisterStateMap = HashMap<u64, FinalRegisterWordState>;

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the REGISTER trace table.
///
/// Creates a table with NUM_REGISTER_ADDRESSES rows (32 regs × 2 Words + 2 for x255 = 66).
/// Each row represents one Word address in register space.
///
/// ## Arguments
///
/// * `final_state` - Map from register Word address to final (timestamp, value)
/// * `entry_point` - ELF entry point address (initial value for x255/PC)
///
/// ## Returns
///
/// The trace table for registers.
pub fn generate_register_trace(
    final_state: &FinalRegisterStateMap,
    entry_point: u64,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    // Rows 0-63: registers x0-x31 (2 Word addresses each)
    for offset in 0..64usize {
        let word_addr = offset as u64;
        let base = offset * cols::NUM_COLUMNS;

        data[base + cols::OFFSET] = FE::from(word_addr);

        // Initial value: all registers start at 0, except SP (x2) which starts at STACK_TOP
        let init_value = if offset == 4 {
            (STACK_TOP & 0xFFFF_FFFF) as u32
        } else if offset == 5 {
            (STACK_TOP >> 32) as u32
        } else {
            0u32
        };
        data[base + cols::INIT] = FE::from(init_value as u64);

        let (timestamp, fini_value) = if let Some(state) = final_state.get(&word_addr) {
            (state.timestamp, state.value)
        } else {
            (0, init_value)
        };

        data[base + cols::FINI] = FE::from(fini_value as u64);
        data[base + cols::TIMESTAMP_LO] = FE::from(timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_HI] = FE::from(timestamp >> 32);
    }

    // Rows 64-65: x255 (PC) at addresses 510, 511
    let entry_point_lo = (entry_point & 0xFFFF_FFFF) as u32;
    let entry_point_hi = (entry_point >> 32) as u32;

    for (i, (addr, init_val)) in [(PC_ADDR_LO, entry_point_lo), (PC_ADDR_HI, entry_point_hi)]
        .iter()
        .enumerate()
    {
        let row = 64 + i;
        let base = row * cols::NUM_COLUMNS;

        data[base + cols::OFFSET] = FE::from(*addr);
        data[base + cols::INIT] = FE::from(*init_val as u64);

        let (timestamp, fini_value) = if let Some(state) = final_state.get(addr) {
            (state.timestamp, state.value)
        } else {
            (0, *init_val)
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
// Preprocessed commitment
// =========================================================================

/// Computes the Merkle root commitment over the LDE of REGISTER precomputed columns.
///
/// The REGISTER table is program-dependent: OFFSET contains register addresses
/// (0..63 for x0-x31, 510-511 for x255/PC), and INIT is 0 except SP (x2)
/// words which hold STACK_TOP and x255 words which hold the entry point.
pub fn compute_precomputed_commitment(options: &ProofOptions, entry_point: u64) -> Commitment {
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();

    let entry_point_lo = entry_point & 0xFFFF_FFFF;
    let entry_point_hi = entry_point >> 32;

    let mut offset_col = vec![FE::zero(); num_rows];
    let mut init_col = vec![FE::zero(); num_rows];

    // Rows 0-63: x0-x31
    for i in 0..64usize {
        offset_col[i] = FE::from(i as u64);
        init_col[i] = if i == 4 {
            FE::from(STACK_TOP & 0xFFFF_FFFF)
        } else if i == 5 {
            FE::from(STACK_TOP >> 32)
        } else {
            FE::zero()
        };
    }

    // Rows 64-65: x255 (PC) at addresses 510, 511
    offset_col[64] = FE::from(PC_ADDR_LO);
    init_col[64] = FE::from(entry_point_lo);
    offset_col[65] = FE::from(PC_ADDR_HI);
    init_col[65] = FE::from(entry_point_hi);

    let columns = [offset_col, init_col];

    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for register column")
        })
        .collect();

    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for register polynomial")
        })
        .collect();

    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    let lde_rows = columns2rows(lde_columns);
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for register LDE");
    tree.root
}

/// Returns the preprocessed commitment for the REGISTER table.
///
/// The commitment is program-dependent (includes entry_point for x255/PC),
/// so it is recomputed each time. The table is tiny (128 rows), so this is fast.
pub fn preprocessed_commitment(options: &ProofOptions, entry_point: u64) -> Commitment {
    compute_precomputed_commitment(options, entry_point)
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
    // Address is just the offset (0..63 for 32 regs × 2 Words, 510-511 for PC)
    // Stored in low word, high word is 0
    let address_lo = BusValue::Packed {
        start_column: cols::OFFSET,
        packing: Packing::Direct,
    };
    let address_hi = BusValue::constant(0);

    vec![
        // REG-C1: memory[1, address, 0, init] - receive initial token
        // Balances MEMW's first send on this address
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
        // Balances MEMW's last receive on this address
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

/// Compute the Word addresses used by a register.
/// Returns 2 addresses: base (low word), base+1 (high word)
pub fn register_word_addresses(reg_idx: u8) -> [u64; 2] {
    let base = register_base_address(reg_idx);
    [base, base + 1]
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
        let trace = generate_register_trace(&final_state, 0x1000);

        // Should have power-of-2 rows >= 66 (32 regs × 2 Words + 2 for x255)
        assert!(trace.num_rows() >= NUM_REGISTER_ADDRESSES);
        assert!(trace.num_rows().is_power_of_two());

        // Check first row (address 0, never accessed)
        assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::INIT), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::FINI), FE::zero());
        assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP_LO), FE::zero());

        // Check x255 (PC) rows
        assert_eq!(*trace.main_table.get(64, cols::OFFSET), FE::from(510u64));
        assert_eq!(
            *trace.main_table.get(64, cols::INIT),
            FE::from(0x1000u64)
        );
        assert_eq!(
            *trace.main_table.get(64, cols::FINI),
            FE::from(0x1000u64)
        ); // fini=init when never accessed
        assert_eq!(*trace.main_table.get(65, cols::OFFSET), FE::from(511u64));
        assert_eq!(*trace.main_table.get(65, cols::INIT), FE::zero()); // entry_point_hi = 0
    }

    #[test]
    fn test_generate_register_trace_with_access() {
        let mut final_state = FinalRegisterStateMap::new();
        // Register x5 low Word was written with value 0x42 at timestamp 100
        let addr = register_base_address(5); // = 10
        final_state.insert(
            addr,
            FinalRegisterWordState {
                timestamp: 100,
                value: 0x42,
            },
        );

        let trace = generate_register_trace(&final_state, 0x1000);

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
