//! GLOBAL_MEMORY table for cross-epoch memory initialization and finalization.
//!
//! The cross-epoch analog of PAGE (`page.rs`): one dense table instance per
//! touched page, bookending the `GlobalMemory` bus that links each epoch's
//! local-to-global (`local_to_global.rs`) boundary claims. For every byte of the
//! page it **sends** a genesis token (the cell's program-start value) and
//! **receives** a finalization token (the cell's value after the last epoch that
//! touched it). Untouched bytes send and receive the identical token, so they
//! cancel — exactly as PAGE's init/fini bookend does on the epoch-local bus.
//!
//! Because the genesis value lives in a PREPROCESSED column (OFFSET + INIT,
//! byte-for-byte identical to PAGE's), the verifier recomputes the same
//! commitment from the ELF via [`page::compute_precomputed_commitment`]. This
//! binds the program's initial memory to the ELF binary.
//!
//! ## Columns
//!
//! | Column | Type | Description |
//! |--------|------|-------------|
//! | offset | RowIndex | 0, 1, ..., page_size-1 (preprocessed) |
//! | init | Byte | Genesis value (from ELF or 0) (preprocessed) |
//! | fini | Byte | Value after the last touching epoch |
//! | fini_epoch | Epoch | Last touching epoch (`GENESIS_EPOCH` if untouched) |
//!
//! Virtual: `address = page_base + offset`, `page_base` constant per instance.
//!
//! ## Bus Interactions
//!
//! GlobalMemory token: `[address_lo, address_hi, value, epoch]` (same order as
//! `local_to_global::bus_interactions`; no timestamp — the chain is ordered by epoch).
//!
//! | Tag | Bus | Token | Multiplicity |
//! |-----|-----|-------|--------------|
//! | GM-GENESIS | GlobalMemory | `[address, init, GENESIS]` | 1 (sender) |
//! | GM-FINAL   | GlobalMemory | `[address, fini, fini_epoch]` | 1 (receiver) |

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};
use stark::trace::TraceTable;

use super::local_to_global::{GENESIS_EPOCH, direct};
use super::page::{DEFAULT_PAGE_SIZE, PageConfig};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices
// =========================================================================

/// Column definitions for the GLOBAL_MEMORY table.
///
/// `address` is virtual, computed as `page_base + offset`; it is NOT a column.
pub mod cols {
    /// offset: Row index (0, 1, ..., page_size-1) - preprocessed
    pub const OFFSET: usize = 0;

    /// init: Genesis byte value (from ELF or 0) - preprocessed
    pub const INIT: usize = 1;

    // Note: there is no init-epoch column. The genesis token always carries
    // `GENESIS_EPOCH`, so the GM-GENESIS sender emits it as a constant (like L2G's
    // `fini_epoch`), saving a column and removing a prover-chosen value.

    /// fini: Final byte value after the last touching epoch
    pub const FINI: usize = 2;

    /// fini_epoch: Last epoch that touched the cell (`GENESIS_EPOCH` if untouched)
    pub const FINI_EPOCH: usize = 3;

    // Note: no fini-timestamp column. The GlobalMemory bus carries no timestamp
    // (the cross-epoch chain is ordered by epoch); timestamps are epoch-local.

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 4;
}

/// Number of preprocessed columns (OFFSET, INIT). Identical to PAGE's preprocessed
/// columns, so the preprocessed commitment is shared with PAGE — compute it with
/// [`page::compute_precomputed_commitment`].
pub const NUM_PREPROCESSED_COLS: usize = 2;

// =========================================================================
// Types
// =========================================================================

/// Final state for a single byte address after the last epoch that touched it.
#[derive(Debug, Clone, Copy, Default)]
pub struct FiniState {
    /// Final byte value.
    pub value: u8,
    /// Index of the last epoch that touched the cell.
    pub epoch: u64,
}

/// Map from byte address to final state, for the bytes touched across all epochs.
pub type FiniStateMap = HashMap<u64, FiniState>;

// =========================================================================
// Trace generation
// =========================================================================

/// Generates a GLOBAL_MEMORY trace for a single page.
///
/// `config` supplies `page_base` and the genesis `init_values` (from the ELF);
/// `final_state` maps each touched byte to its final value and last-touch epoch.
pub fn generate_global_trace(
    config: &PageConfig,
    final_state: &FiniStateMap,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let page_size = DEFAULT_PAGE_SIZE;
    let page_base = config.page_base;

    assert!(
        page_base.is_multiple_of(page_size as u64),
        "Page base must be page-aligned"
    );

    let num_rows = page_size; // One row per byte in the page
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for offset in 0..page_size {
        let byte_addr = page_base + (offset as u64);
        let base = offset * cols::NUM_COLUMNS;

        // Offset (preprocessed) - address is virtual: page_base + offset
        data[base + cols::OFFSET] = FE::from(offset as u64);

        // Genesis value (init_values may be shorter than the page → trailing zeros)
        let init_value = config
            .init_values
            .as_ref()
            .and_then(|v| v.get(offset).copied())
            .unwrap_or(0);
        data[base + cols::INIT] = FE::from(init_value as u64);

        // Final state: if touched use it, otherwise the cell stays at genesis
        // (fini=init, epoch=GENESIS) so its genesis/finalization tokens cancel.
        let (fini_value, fini_epoch) = match final_state.get(&byte_addr) {
            Some(state) => (state.value, state.epoch),
            None => (init_value, GENESIS_EPOCH),
        };

        data[base + cols::FINI] = FE::from(fini_value as u64);
        data[base + cols::FINI_EPOCH] = FE::from(fini_epoch);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates the GlobalMemory bus interactions for a GLOBAL_MEMORY table.
///
/// The token order matches `local_to_global::bus_interactions` exactly:
/// `[address_lo, address_hi, value, epoch]` (no timestamp — the cross-epoch chain
/// is ordered by epoch). The address is computed as `page_base + offset` via a
/// linear combination, like PAGE.
///
/// - GM-GENESIS: sends `[address, init, GENESIS]` — the token an L2G
///   init-receiver consumes for a genesis-origin cell.
/// - GM-FINAL: receives `[address, fini, fini_epoch]` — the token the
///   last touching epoch's L2G fini-sender produces.
pub fn bus_interactions(page_base: u64) -> Vec<BusInteraction> {
    let page_base_lo = page_base & 0xFFFF_FFFF;
    let page_base_hi = page_base >> 32;

    let address_lo = BusValue::linear(vec![
        LinearTerm::Constant(page_base_lo as i64),
        LinearTerm::Column {
            coefficient: 1,
            column: cols::OFFSET,
        },
    ]);
    let address_hi = BusValue::constant(page_base_hi);

    vec![
        // GM-GENESIS: send the genesis token [address, init, GENESIS]. No timestamp:
        // the GlobalMemory chain is ordered by epoch (timestamps are epoch-local).
        BusInteraction::sender(
            BusId::GlobalMemory,
            Multiplicity::One,
            vec![
                address_lo.clone(),
                address_hi.clone(),
                direct(cols::INIT),
                BusValue::constant(GENESIS_EPOCH),
            ],
        ),
        // GM-FINAL: receive the finalization token [address, fini, fini_epoch].
        // Note: FINI has no explicit AreBytes range check here (unlike PAGE's fini).
        // It's byte-checked transitively: this receiver must match an L2G fini token
        // on the GlobalMemory bus, and L2G already AreBytes-checks its fini value. So
        // a non-byte FINI could never balance. Do not "add a missing AreBytes" here.
        BusInteraction::receiver(
            BusId::GlobalMemory,
            Multiplicity::One,
            vec![
                address_lo,
                address_hi,
                direct(cols::FINI),
                direct(cols::FINI_EPOCH),
            ],
        ),
    ]
}
