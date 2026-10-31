//! REGISTER table for register initialization and finalization.
//!
//! Similar to PAGE table but for registers (is_register=1).
//! Provides initial and final tokens for the Memory bus to balance
//! register read/write operations from MEMW.
//!
//! ## Token Model
//!
//! - **REG-C1**: Receives initial token `(1, address, ts=1, init)` - balances MEMW's send on first access
//! - **REG-C2**: Sends final token `(1, address, timestamp, fini)` - balances MEMW's receive on last access.
//!   For the PC (x255) the timestamp is the verifier-known `final_pc_ts` instead
//!   of the `timestamp` column (spec `halt.typ`).
//!
//! ## Columns
//!
//! | Column | Type | Description |
//! |--------|------|-------------|
//! | offset | RowIndex | Word address within register space (preprocessed) |
//! | init | Word | Initial value (preprocessed) |
//! | fini | Word | Final value (preprocessed) |
//! | is_pc | Bit | 1 on the two x255 rows (preprocessed) |
//! | timestamp | DWordWL | Final timestamp (1 if never accessed; unused on x255 rows) |
//!
//! OFFSET, INIT, FINI and IS_PC are preprocessed, so the verifier knows both register
//! files. A monolithic proof (and the final continuation epoch) pins FINI to the
//! halted state ([`halted_fini`]); a non-final epoch's FINI is the next epoch's INIT.

use std::collections::HashMap;

use math::polynomial::Polynomial;
use stark::config::Commitment;
use stark::leaf_layout::LeafLayout;
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

/// Every register access is 64-bit = 2 Words of 32 bits each, x254 included: the
/// COPY chip reads it through the same `REG` template as x10, so the two share one
/// interaction and therefore one access width.
pub const WORDS_PER_REGISTER: usize = 2;

/// Total number of register Word addresses.
/// x0-x31 use addresses 0..63, x254 uses 508..509, x255 uses addresses 510..511.
pub const NUM_REGISTER_ADDRESSES: usize = NUM_REGISTERS * WORDS_PER_REGISTER;

/// Number of preprocessed columns (OFFSET, INIT, FINI, IS_PC). OFFSET encodes the
/// Word address, INIT the initial value and FINI the final one, so both register
/// files are verifier-known public values: the verifier recomputes the commitment
/// from them and the REG-C2 Memory-bus token forces FINI to equal the true final
/// registers. IS_PC selects the x255 rows, whose final token REG-C2 sends at the
/// verifier-known final PC timestamp.
///
/// A continuation epoch preprocesses FINI = `R_{i+1}` and the next epoch reuses it
/// as its INIT, binding `init(epoch i+1) == fini(epoch i)` with no extra bus. A
/// monolithic proof (and the final epoch) pins FINI to the halted state (see
/// [`halted_fini`]).
pub const NUM_PREPROCESSED_COLS: usize = 4;

// =========================================================================
// Column indices for REGISTER table
// =========================================================================

pub mod cols {
    /// offset: Row index / byte address within register space
    pub const OFFSET: usize = 0;

    /// init: Initial Word value (preprocessed)
    pub const INIT: usize = 1;

    /// fini: Final Word value (preprocessed)
    pub const FINI: usize = 2;

    /// is_pc: 1 on the x255 (PC) rows, 0 elsewhere (preprocessed)
    pub const IS_PC: usize = 3;

    /// timestamp[0]: Final timestamp low word (1 if never accessed, matching REG-C1 init)
    pub const TIMESTAMP_LO: usize = 4;

    /// timestamp[1]: Final timestamp high word
    pub const TIMESTAMP_HI: usize = 5;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 6;
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
/// x0-x31 use addresses 0..63, x254 uses 508..509, x255 uses 510..511.
pub(crate) fn register_word_address_list() -> [u64; NUM_REGISTER_ADDRESSES] {
    let mut addrs = [0u64; NUM_REGISTER_ADDRESSES];
    // x0-x31: addresses 0..63
    for (i, addr) in addrs.iter_mut().enumerate().take(64) {
        *addr = i as u64;
    }
    // x254: synthetic commit index, addresses 508, 509
    addrs[64] = 508;
    addrs[65] = 509;
    // x255: addresses 510, 511
    addrs[66] = 510;
    addrs[67] = 511;
    addrs
}

// Positions of the non-general-purpose registers within a register-init vector
// (indexed in `register_word_address_list` order). x0-x31 occupy positions 0..63
// (position `i` is word address `i`), so register `r`'s two words are at `2r`, `2r+1`.
/// Position of x254 (synthetic commit index) low word (word address 508).
pub(crate) const X254_INDEX: usize = 64;
/// Position of x254 high word (word address 509).
#[cfg(test)]
pub(crate) const X254_HI_INDEX: usize = 65;
/// Position of x255 (PC) low word (word address 510).
pub(crate) const PC_LO_INDEX: usize = 66;
/// Position of x255 (PC) high word (word address 511).
pub(crate) const PC_HI_INDEX: usize = 67;

/// Compute the initial value for a register Word address.
///
/// This is the **program-start** register image, so it only applies to the first
/// continuation epoch (or a whole-program run). Later epochs start mid-execution
/// and supply their own boundary register snapshot instead.
///
/// - SP (x2) words at offset 4,5 hold STACK_TOP
/// - x254 at offsets 508,509 is the synthetic commit index, initialized to 0
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

/// The final register file of every run that halts, as the verifier knows it
/// (`fini_from_trace` order): HALT zeroizes x1-x31 (x10 is read as 0), x0 is never
/// written, x254 counts the committed bytes, and the PC ends on the padding value
/// `pc = 1`.
///
/// Pinning FINI is what stops the PC chain from ending anywhere but `pc = 1`.
/// HALT's `pc` column is free, so with a free FINI a prover could cancel HALT's
/// consume/emit and keep executing past the exit ecall, ending the chain on the
/// last post-exit `next_pc` (see `tests/halt_pc_forgery_poc.rs`).
///
/// The final PC timestamp is pinned too (REG-C2 consumes x255 at the
/// verifier-known `final_pc_ts`, spec `halt.typ`), but the value alone already
/// suffices: no real row writes `next_pc = 1` (instruction addresses are even
/// and BRANCH clears the LSB), so a token with value 1 can only come from
/// HALT's emit or a padding row after it.
pub fn halted_fini(committed_bytes: u32) -> Vec<u32> {
    let mut fini = vec![0u32; NUM_REGISTER_ADDRESSES];
    fini[X254_INDEX] = committed_bytes;
    fini[PC_LO_INDEX] = super::cpu::CPU_PADDING_PC as u32;
    fini
}

/// REGISTER preprocessed commitment for a monolithic proof: OFFSET, INIT from the
/// entry point, and FINI pinned to [`halted_fini`].
pub fn halted_commitment(
    options: &ProofOptions,
    entry_point: u64,
    committed_bytes: u32,
) -> Commitment {
    let init = register_init_from_entry_point(entry_point);
    compute_precomputed_commitment(options, &init, &halted_fini(committed_bytes))
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
    // is only correct for an epoch with no preceding `write`. The production path
    // carries x254 across epochs via the previous epoch's bound FINI vector, not
    // this snapshot helper.
    init[X254_INDEX] = 0;
    init[X254_HI_INDEX] = 0;
    init[PC_LO_INDEX] = (pc & 0xFFFF_FFFF) as u32;
    init[PC_HI_INDEX] = (pc >> 32) as u32;
    init
}

/// Generates the REGISTER trace table.
///
/// Creates a table with NUM_REGISTER_ADDRESSES rows.
/// Each row represents one Word address in register space.
/// x0-x31 at addresses 0..63, x254 at 508..509, x255 (PC) at 510..511.
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
        table.set_bool(row, cols::IS_PC, is_pc_address(word_addr));
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

/// The preprocessed columns themselves: OFFSET (the Word address: 0..63 for
/// x0-x31, 508-509 for x254, 510-511 for x255), INIT, FINI and IS_PC, padded to a
/// power of two. FINI and IS_PC are 0 on padding rows, as the trace builds them.
///
/// ★ THE COLUMNS, NOT ONLY THEIR ROOT, because the two verifiers bind different
/// things. The univariate verifier compares [`compute_precomputed_commitment`]'s
/// root against the proof's precomputed tree (`stark::verifier`, the
/// `is_preprocessed()` branch). The multilinear verifier has no such tree: it
/// checks the claimed openings of columns `0..num_precomputed_columns()` against
/// the values the program implies (`stark::multilinear_table::check_preprocessed`),
/// and it reads them through [`stark::traits::AIR::precomputed_columns`]. An AIR
/// built with a commitment and no columns closure hands that check an EMPTY list,
/// which it walks in zero iterations — so this function is what makes INIT and
/// FINI binding on the multilinear path at all.
///
/// `fini` is the vector produced by `fini_from_trace` (entry `i` = the register
/// at `register_word_address_list()[i]`): a continuation epoch's `R_{i+1}`, or a
/// monolithic proof's (and a block's) halted file ([`halted_fini`]).
///
/// # ★ Why the epoch statement does not absorb `init` or `fini`, and must not
///
/// The chain of custody is already closed without an absorb. The commitment
/// ROOTS go into the transcript before any challenge is drawn, so the FINI
/// column is fixed before the prover learns the reduced point; `check_preprocessed`
/// then binds the verifier's own `(init, fini)` to that committed column AT that
/// point. A prover therefore cannot choose the claim after the fact: the column
/// is committed first and the value it is compared against is the verifier's,
/// not the proof's. Adding these vectors to `absorb_epoch` would bind nothing
/// further and would move every pinned transcript constant.
pub fn preprocessed_columns(init: &[u32], fini: &[u32]) -> Vec<Vec<FE>> {
    debug_assert_eq!(fini.len(), NUM_REGISTER_ADDRESSES);
    let num_rows = NUM_REGISTER_ADDRESSES.next_power_of_two();
    let addr_list = register_word_address_list();

    let mut offset_col = crate::tables::types::zeroed_fe_vec(num_rows);
    let mut init_col = crate::tables::types::zeroed_fe_vec(num_rows);
    let mut fini_col = crate::tables::types::zeroed_fe_vec(num_rows);
    let mut is_pc_col = crate::tables::types::zeroed_fe_vec(num_rows);

    for i in 0..NUM_REGISTER_ADDRESSES {
        offset_col[i] = FE::from(addr_list[i]);
        init_col[i] = FE::from(init.get(i).copied().unwrap_or(0) as u64);
        fini_col[i] = FE::from(fini.get(i).copied().unwrap_or(0) as u64);
        is_pc_col[i] = FE::from(u64::from(is_pc_address(addr_list[i])));
    }

    vec![offset_col, init_col, fini_col, is_pc_col]
}

/// Computes the Merkle root commitment over the LDE of the REGISTER preprocessed
/// columns ([`preprocessed_columns`]), so the univariate verifier recomputes it
/// from the public `init` and `fini` and the proof's FINI column is locked to
/// `fini`. The multilinear path never reads this root.
pub fn compute_precomputed_commitment(
    options: &ProofOptions,
    init: &[u32],
    fini: &[u32],
) -> Commitment {
    compute_precomputed_commitment_with(options, init, fini, LeafLayout::RowPair)
}

/// [`compute_precomputed_commitment`] under an explicit leaf layout (S2) — the
/// host twin of the in-circuit register commitment
/// (`lfm::programs::emit_register_commitment` at the same `rows_per_leaf`).
pub fn compute_precomputed_commitment_with(
    options: &ProofOptions,
    init: &[u32],
    fini: &[u32],
    layout: LeafLayout,
) -> Commitment {
    commit_register_columns(options, preprocessed_columns(init, fini), layout)
}

/// LDE + bit-reverse + Merkle-commit the given preprocessed columns (in column
/// order).
fn commit_register_columns(
    options: &ProofOptions,
    columns: Vec<Vec<FE>>,
    layout: LeafLayout,
) -> Commitment {
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

    // ★ Through the LFM commit helper, which commits under the BLOCK PATH's pin
    // rather than `stark`'s default aliases. This root is a PREPROCESSED
    // commitment the prover recomputes and compares against, so building it with
    // a different hash than the path commits under fails at prove time with
    // `PrecomputedCommitmentMismatch` — which is exactly how it was found.
    crate::lfm::commit::commit_lde_columns_with(&lde_columns, layout)
}

/// REGISTER's commitment source for both leaf layouts, from `init` and `fini`:
/// the row-pair root `row_pair` the caller holds, and the one-row root computed
/// from the same files on first use.
pub fn lazy_commitment(
    options: &ProofOptions,
    row_pair: Commitment,
    init: &[u32],
    fini: &[u32],
) -> stark::lookup::LazyCommitment {
    let (o, init, fini) = (options.clone(), init.to_vec(), fini.to_vec());
    stark::lookup::LazyCommitment::ready(row_pair).with_one_row(move || {
        Some(compute_precomputed_commitment_with(
            &o,
            &init,
            &fini,
            LeafLayout::Row,
        ))
    })
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the REGISTER table.
///
/// ## Bus Interactions
///
/// - REG-C1: memory[1, address, 1, init] - receiver, multiplicity -1
/// - REG-C2: memory[1, address, timestamp, fini] - sender, multiplicity `1 - is_pc`
/// - REG-C2 (PC): memory[1, address, final_pc_ts, fini] - sender, multiplicity `is_pc`
///
/// `final_pc_ts` is where the last CPU row writes `next_pc`
/// ([`super::cpu::final_pc_timestamp`]). The verifier derives it from the CPU
/// table heights, so the PC chain's end is verifier-known (spec `halt.typ`).
///
/// Note: is_register=1 (constant) to distinguish from memory (is_register=0).
pub fn bus_interactions(final_pc_ts: u64) -> Vec<BusInteraction> {
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
        // REG-C2 (PC): memory[1, address, final_pc_ts, fini] - send the final PC
        // token at the verifier-known timestamp. Balances the last CPU row's
        // inline-PC write.
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Column(cols::IS_PC),
            vec![
                BusValue::constant(1),
                address_lo.clone(),
                address_hi.clone(),
                BusValue::constant(final_pc_ts),
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: cols::FINI,
                    packing: Packing::Direct,
                },
            ],
        ),
        // REG-C2: memory[1, address, timestamp, fini] - send final token
        // Balances MEMW's last receive on this address
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Negated(cols::IS_PC),
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

/// Whether `word_addr` is one of the two x255 (PC) Word addresses.
pub(crate) fn is_pc_address(word_addr: u64) -> bool {
    word_addr == 510 || word_addr == 511
}

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
        254 => vec![508, 509],
        255 => vec![510, 511],
        _ => {
            let base = 2 * reg_idx as u64;
            vec![base, base + 1]
        }
    }
}
