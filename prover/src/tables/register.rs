//! REGISTER table for register initialization and finalization.
//!
//! Similar to PAGE table but for registers (is_register=1).
//! Provides initial and final tokens for the Memory bus to balance
//! register read/write operations from MEMW.
//!
//! ## Token Model
//!
//! - **REG-C1**: Receives initial token `(1, address, ts=1, init)` - balances MEMW's send on first access
//! - **REG-C2**: Sends final token `(1, address, timestamp, fini)` - balances MEMW's receive on last access
//!
//! ## Columns
//!
//! | Column | Type | Description |
//! |--------|------|-------------|
//! | offset | RowIndex | Byte offset within register space |
//! | init | Word | Initial value (0 for all registers at start) |
//! | fini | Word | Final value after execution |
//! | timestamp | DWordWL | Final timestamp (1 if never accessed) |

use std::collections::HashMap;

use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed};
use stark::config::Commitment;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::TraceTable;

use super::page::STACK_TOP;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Constants
// =========================================================================

/// Number of logical registers represented in the table:
/// x0-x31 (32 GPRs), x254 (synthetic commit index), x255 (PC register).
pub const NUM_REGISTERS: usize = 34;

/// Most register accesses are 64-bit = 2 Words of 32 bits each.
/// The COMMIT spec adds a synthetic single-word x254 entry at address 508.
pub const WORDS_PER_REGISTER: usize = 2;

/// Total number of register Word addresses.
/// x0-x31 use addresses 0..63, x254 uses address 508, x255 uses addresses 510..511.
/// -1 because x254 is single-word (1 address instead of 2).
pub const NUM_REGISTER_ADDRESSES: usize = NUM_REGISTERS * WORDS_PER_REGISTER - 1;

/// Number of preprocessed columns (OFFSET, INIT).
/// OFFSET encodes the Word address, INIT holds the initial value.
/// Program-dependent: x255 init = ELF entry point.
pub const NUM_PREPROCESSED_COLS: usize = 2;

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

    /// timestamp[0]: Final timestamp low word (1 if never accessed, matching REG-C1 init)
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
    /// Final timestamp (1 if never accessed, matching REG-C1 init)
    pub timestamp: u64,
    /// Final Word value (32-bit)
    pub value: u32,
}

/// Map from register Word address to final state.
pub type FinalRegisterStateMap = HashMap<u64, FinalRegisterWordState>;

// =========================================================================
// Trace generation
// =========================================================================

/// Returns the Word addresses for all register table rows.
///
/// x0-x31 use addresses 0..63, x254 uses address 508, x255 uses 510..511.
fn register_word_address_list() -> [u64; NUM_REGISTER_ADDRESSES] {
    let mut addrs = [0u64; NUM_REGISTER_ADDRESSES];
    // x0-x31: addresses 0..63
    for (i, addr) in addrs.iter_mut().enumerate().take(64) {
        *addr = i as u64;
    }
    // x254: synthetic commit index (single-word)
    addrs[64] = 508;
    // x255: addresses 510, 511
    addrs[65] = 510;
    addrs[66] = 511;
    addrs
}

/// Compute the initial value for a register Word address.
///
/// - SP (x2) words at offset 4,5 hold STACK_TOP
/// - x254 at offset 508 is the synthetic commit index, initialized to 0
/// - PC (x255) words at offset 510,511 hold entry_point
/// - All others are 0
fn init_value_for_address(word_addr: u64, entry_point: u64) -> u32 {
    match word_addr {
        4 => (STACK_TOP & 0xFFFF_FFFF) as u32,
        5 => (STACK_TOP >> 32) as u32,
        510 => (entry_point & 0xFFFF_FFFF) as u32,
        511 => (entry_point >> 32) as u32,
        _ => 0,
    }
}

/// Generates the REGISTER trace table.
///
/// Creates a table with NUM_REGISTER_ADDRESSES rows.
/// Each row represents one Word address in register space.
/// x0-x31 at addresses 0..63, x254 at 508, x255 (PC) at 510..511.
///
/// ## Arguments
///
/// * `final_state` - Map from register Word address to final (timestamp, value)
/// * `entry_point` - ELF entry point (initial PC value for x255)
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
    let addr_list = register_word_address_list();

    for (row, &word_addr) in addr_list.iter().enumerate().take(NUM_REGISTER_ADDRESSES) {
        let base = row * cols::NUM_COLUMNS;

        // Offset = actual Word address in register space
        data[base + cols::OFFSET] = FE::from(word_addr);

        let init_value = init_value_for_address(word_addr, entry_point);
        data[base + cols::INIT] = FE::from(init_value as u64);

        // Final state: if accessed use final, otherwise use initial (timestamp 1)
        let (timestamp, fini_value) = if let Some(state) = final_state.get(&word_addr) {
            (state.timestamp, state.value)
        } else {
            // Never accessed: timestamp=1 (matches REG-C1 init), fini=init
            (1, init_value)
        };

        data[base + cols::FINI] = FE::from(fini_value as u64);
        data[base + cols::TIMESTAMP_LO] = FE::from(timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_HI] = FE::from(timestamp >> 32);
    }

    // Padding rows (if num_rows > NUM_REGISTER_ADDRESSES): set TIMESTAMP_LO=1 so
    // REG-C1's constant ts=1 emission matches REG-C2's ts=TIMESTAMP_LO consumption,
    // keeping padding rows self-cancelling on the bus.
    for row in NUM_REGISTER_ADDRESSES..num_rows {
        let base = row * cols::NUM_COLUMNS;
        data[base + cols::TIMESTAMP_LO] = FE::from(1u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

/// Computes the Merkle root commitment over the LDE of REGISTER precomputed columns.
///
/// Program-dependent: x255 (PC) init = entry_point.
/// OFFSET encodes the Word address (0..63 for x0-x31, 508 for x254, 510-511 for x255).
/// INIT holds the initial value (SP=STACK_TOP, PC=entry_point, rest=0).
pub fn compute_precomputed_commitment(options: &ProofOptions, entry_point: u64) -> Commitment {
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
    let addr_list = register_word_address_list();

    let mut offset_col = vec![FE::zero(); num_rows];
    let mut init_col = vec![FE::zero(); num_rows];

    for i in 0..NUM_REGISTER_ADDRESSES {
        let word_addr = addr_list[i];
        offset_col[i] = FE::from(word_addr);
        init_col[i] = FE::from(init_value_for_address(word_addr, entry_point) as u64);
    }

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
    let lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for register polynomial")
        })
        .collect();

    let (_, root) = commit_bit_reversed(&lde_columns, ROWS_PER_LEAF)
        .expect("Failed to build Merkle tree for register LDE");
    root
}

/// Returns the preprocessed commitment for the REGISTER table.
///
/// Program-dependent (entry_point varies per ELF), so not globally cached.
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
/// - REG-C1: memory[1, address, 1, init] - receiver, multiplicity -1
/// - REG-C2: memory[1, address, timestamp, fini] - sender, multiplicity 1
///
/// Note: is_register=1 (constant) to distinguish from memory (is_register=0).
pub fn bus_interactions() -> Vec<BusInteraction> {
    // Address is just the offset in register space.
    // Stored in low word, high word is 0
    let address_lo = BusValue::Packed {
        start_column: cols::OFFSET,
        packing: Packing::Direct,
    };
    let address_hi = BusValue::constant(0);

    vec![
        // REG-C1: memory[1, address, 1, init] - receive initial token
        // Balances MEMW's first send on this address.
        // Per spec/memory.typ: "register initialization happens at timestamp 1"
        // so that the CPU's inline PC read on the first row consumes the init token.
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
                // timestamp_lo = 1 (initial)
                BusValue::constant(1),
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
pub fn register_base_address(reg_idx: u8) -> u64 {
    match reg_idx {
        254 => 508,
        255 => 510,
        _ => 2 * reg_idx as u64,
    }
}

/// Compute the Word addresses used by a register.
pub fn register_word_addresses(reg_idx: u8) -> Vec<u64> {
    match reg_idx {
        254 => vec![508],
        255 => vec![510, 511],
        _ => {
            let base = 2 * reg_idx as u64;
            vec![base, base + 1]
        }
    }
}
