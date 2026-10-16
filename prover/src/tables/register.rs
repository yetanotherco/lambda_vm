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

#[cfg(test)]
use executor::vm::registers::Registers;

use super::page::STACK_TOP;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

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

/// Number of preprocessed columns (OFFSET, INIT) for the monolithic prover.
/// OFFSET encodes the Word address, INIT holds the initial value.
/// Program-dependent: x255 init = ELF entry point.
pub const NUM_PREPROCESSED_COLS: usize = 2;

/// Number of preprocessed columns (OFFSET, INIT, FINI) for continuation epochs.
/// A continuation epoch additionally preprocesses FINI so the epoch's final
/// register file becomes a verifier-known public value (`R_{i+1}`): the verifier
/// recomputes the commitment from it, the REG-C2 Memory-bus token forces it to
/// equal the true final registers, and the next epoch reuses the same `R_{i+1}`
/// as its preprocessed INIT — binding `init(epoch i+1) == fini(epoch i)` with no
/// extra bus. The monolithic prover keeps FINI as a main-trace column (it has no
/// verifier-known final state), using `NUM_PREPROCESSED_COLS` instead.
pub const NUM_PREPROCESSED_COLS_WITH_FINI: usize = 3;

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
pub(crate) fn register_word_address_list() -> [u64; NUM_REGISTER_ADDRESSES] {
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

// Positions of the non-general-purpose registers within a register-init vector
// (indexed in `register_word_address_list` order). x0-x31 occupy positions 0..63
// (position `i` is word address `i`), so register `r`'s two words are at `2r`, `2r+1`.
/// Position of x254 (synthetic commit index, word address 508).
pub(crate) const X254_INDEX: usize = 64;
/// Position of x255 (PC) low word (word address 510).
pub(crate) const PC_LO_INDEX: usize = 65;
/// Position of x255 (PC) high word (word address 511).
pub(crate) const PC_HI_INDEX: usize = 66;

/// Compute the initial value for a register Word address.
///
/// This is the **program-start** register image, so it only applies to the first
/// continuation epoch (or a whole-program run). Later epochs start mid-execution
/// and supply their own boundary register snapshot instead.
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

/// Build the register init vector (one initial value per row, in
/// `register_word_address_list` order) for a program starting at `entry_point`
/// (the program-start register image). A continuation epoch would instead supply
/// its boundary register snapshot.
pub(crate) fn register_init_from_entry_point(entry_point: u64) -> Vec<u32> {
    register_word_address_list()
        .iter()
        .map(|&addr| init_value_for_address(addr, entry_point))
        .collect()
}

/// Build the register init map from an epoch's boundary register snapshot: the
/// executor `Registers` (x1-x31, including SP) plus the program counter (x255).
/// x0 and the synthetic commit index (x254) are zero in the naive version.
///
/// Used by tests that build a single epoch from a boundary snapshot. The
/// continuation prover no longer uses this for chaining: epoch i+1's register
/// init comes from epoch i's *bound* fini (`fini_from_final_state`, the same
/// value the trace binds — pinned by `fini_from_final_state_matches_trace` —
/// carried as the next epoch's preprocessed INIT), not a trusted snapshot.
#[cfg(test)]
pub(crate) fn register_init_from_snapshot(registers: &Registers, pc: u64) -> Vec<u32> {
    let mut init = vec![0u32; NUM_REGISTER_ADDRESSES];
    for reg in 0u8..32 {
        let value = if reg == 0 {
            0
        } else {
            registers.read(reg as u32).unwrap_or(0)
        };
        let base = (reg as usize) * 2;
        init[base] = (value & 0xFFFF_FFFF) as u32;
        init[base + 1] = (value >> 32) as u32;
    }
    // x254 synthetic commit index, hardcoded to 0 in this test-only helper, so it
    // is only correct for an epoch with no preceding COMMIT. The production path
    // carries x254 across epochs via the previous epoch's bound FINI vector, not
    // this snapshot helper.
    init[X254_INDEX] = 0;
    init[PC_LO_INDEX] = (pc & 0xFFFF_FFFF) as u32;
    init[PC_HI_INDEX] = (pc >> 32) as u32;
    init
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
/// * `init` - Initial value per row, in `register_word_address_list` order
///   (program-start image, or an epoch's boundary register snapshot)
///
/// ## Returns
///
/// The trace table for registers.
pub fn generate_register_trace(
    final_state: &FinalRegisterStateMap,
    init: &[u32],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;
    let addr_list = register_word_address_list();

    for (row, &word_addr) in addr_list.iter().enumerate().take(NUM_REGISTER_ADDRESSES) {
        // Offset = actual Word address in register space
        table.set_u64(row, cols::OFFSET, word_addr);

        let init_value = init.get(row).copied().unwrap_or(0);
        table.set_word(row, cols::INIT, init_value);

        // Final state: if accessed use final, otherwise use initial (timestamp 1)
        let (timestamp, fini_value) = if let Some(state) = final_state.get(&word_addr) {
            (state.timestamp, state.value)
        } else {
            // Never accessed: timestamp=1 (matches REG-C1 init), fini=init
            (1, init_value)
        };

        table.set_word(row, cols::FINI, fini_value);
        table.set_dword_wl(row, cols::TIMESTAMP_LO, timestamp);
    }

    // Padding rows (if num_rows > NUM_REGISTER_ADDRESSES): set TIMESTAMP_LO=1 so
    // REG-C1's constant ts=1 emission matches REG-C2's ts=TIMESTAMP_LO consumption,
    // keeping padding rows self-cancelling on the bus.
    for row in NUM_REGISTER_ADDRESSES..num_rows {
        table.set_u64(row, cols::TIMESTAMP_LO, 1);
    }

    trace
}

/// Extract the per-register final values (`R_{i+1}`) from a committed REGISTER
/// trace: reads `FINI` on the real rows (the first `NUM_REGISTER_ADDRESSES`) into
/// a vector in `register_word_address_list` order — entry `i` is the final value
/// of the register at `register_word_address_list()[i]`. This is the epoch's final
/// register file; the continuation builds this epoch's preprocessed FINI
/// commitment from it and reuses it as the next epoch's preprocessed INIT.
pub fn fini_from_trace(trace: &TraceTable<GoldilocksField, GoldilocksExtension>) -> Vec<u32> {
    (0..NUM_REGISTER_ADDRESSES)
        .map(|row| trace.main_table.get(row, cols::FINI).to_raw() as u32)
        .collect()
}

/// [`fini_from_trace`] without the trace: derives the same final register file
/// directly from the collected final state, mirroring exactly how
/// [`generate_register_trace`] fills the `FINI` column (accessed registers take
/// their final value; never-accessed registers keep their init). Lets the
/// continuation producer chain epochs before this epoch's REGISTER trace is
/// generated. Pinned to the trace-derived values by
/// `fini_from_final_state_matches_trace`.
pub fn fini_from_final_state(final_state: &FinalRegisterStateMap, init: &[u32]) -> Vec<u32> {
    register_word_address_list()
        .iter()
        .take(NUM_REGISTER_ADDRESSES)
        .enumerate()
        .map(|(row, word_addr)| {
            final_state
                .get(word_addr)
                .map(|state| state.value)
                .unwrap_or_else(|| init.get(row).copied().unwrap_or(0))
        })
        .collect()
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

/// Computes the Merkle root commitment over the LDE of REGISTER precomputed columns.
///
/// Program-dependent: x255 (PC) init = entry_point.
/// OFFSET encodes the Word address (0..63 for x0-x31, 508 for x254, 510-511 for x255).
/// INIT holds the initial value (SP=STACK_TOP, PC=entry_point, rest=0).
pub fn compute_precomputed_commitment(options: &ProofOptions, init: &[u32]) -> Commitment {
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
    let addr_list = register_word_address_list();

    let mut offset_col = crate::tables::types::zeroed_fe_vec(num_rows);
    let mut init_col = crate::tables::types::zeroed_fe_vec(num_rows);

    for i in 0..NUM_REGISTER_ADDRESSES {
        offset_col[i] = FE::from(addr_list[i]);
        init_col[i] = FE::from(init.get(i).copied().unwrap_or(0) as u64);
    }

    commit_register_columns(options, vec![offset_col, init_col])
}

/// Continuation variant: commits OFFSET + INIT + FINI, so the verifier recomputes
/// the commitment from the public `init` (`R_i`) and `fini` (`R_{i+1}`) and the
/// proof's FINI column is locked to `R_{i+1}`. `fini` is the vector produced by
/// `fini_from_trace` (entry `i` = the register at `register_word_address_list()[i]`).
/// Used by continuation epochs with `NUM_PREPROCESSED_COLS_WITH_FINI`; must match
/// the column order of the REGISTER trace (OFFSET, INIT, FINI), and FINI on padding
/// rows is 0 (as the trace builds it).
pub fn compute_precomputed_commitment_with_fini(
    options: &ProofOptions,
    init: &[u32],
    fini: &[u32],
) -> Commitment {
    debug_assert_eq!(fini.len(), NUM_REGISTER_ADDRESSES);
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
    let addr_list = register_word_address_list();

    let mut offset_col = crate::tables::types::zeroed_fe_vec(num_rows);
    let mut init_col = crate::tables::types::zeroed_fe_vec(num_rows);
    let mut fini_col = crate::tables::types::zeroed_fe_vec(num_rows);

    for i in 0..NUM_REGISTER_ADDRESSES {
        offset_col[i] = FE::from(addr_list[i]);
        init_col[i] = FE::from(init.get(i).copied().unwrap_or(0) as u64);
        fini_col[i] = FE::from(fini[i] as u64);
    }

    commit_register_columns(options, vec![offset_col, init_col, fini_col])
}

/// LDE + bit-reverse + Merkle-commit the given preprocessed columns (in column
/// order). Shared by the monolithic (OFFSET, INIT) and continuation
/// (OFFSET, INIT, FINI) preprocessed commitments.
fn commit_register_columns(options: &ProofOptions, columns: Vec<Vec<FE>>) -> Commitment {
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
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
pub fn preprocessed_commitment(options: &ProofOptions, init: &[u32]) -> Commitment {
    compute_precomputed_commitment(options, init)
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
