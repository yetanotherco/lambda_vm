//! GLOBAL_FIELD_MEMORY table for cross-epoch field-storage init/finalization.
//!
//! The field-storage (memory domains 3/4/5) analog of `global_memory.rs`
//! (`GLOBAL_MEMORY`, RAM): it closes the two loose ends of the per-cell cross-epoch
//! chain that FEXT_PAGE's continuation-mode boundary claims open on the
//! `GlobalFieldMemory` bus. For every field-storage cell `(domain, addr)` touched
//! anywhere in the run it **sends** a genesis token (value 0 — field-storage is
//! zero-initialized) and **receives** a finalization token (the cell's value after
//! the last epoch that touched it).
//!
//! ## Why sparse (not dense like `GLOBAL_MEMORY`)
//!
//! RAM's `GLOBAL_MEMORY` is dense — one preprocessed row per byte of a page — so
//! `(address)` uniqueness is automatic. Field-storage lives in a 64-bit address
//! space across three domains, which cannot be enumerated densely, so this table is
//! **sparse**: one committed row per touched cell. Uniqueness of `(domain, addr)`
//! must therefore be enforced explicitly, with the same sorted-keys argument
//! FEXT_PAGE uses (see `fext_page.rs`). Without it a prover could emit two genesis
//! tokens for one cell and let a later epoch re-read 0 — an unsound mid-run reset.
//!
//! ## Soundness sketch
//!
//! The `GlobalFieldMemory` bus balances over the whole global proof iff, per cell:
//! - exactly one genesis send `[domain, addr, 0, GENESIS]` (this table, value pinned
//!   to the constant 0), consumed by the first-touching epoch's init receiver
//!   (init_epoch = GENESIS);
//! - the telescoping `fini(epoch i) == init(epoch i+1)` holds (FEXT_PAGE tokens);
//! - exactly one final receive `[domain, addr, fini_val, fini_epoch]` (this table),
//!   consuming the last-touching epoch's fini send.
//!
//! Completeness (a row per touched cell) is forced by the bus: an omitted genesis
//! dangles the first init receiver; an omitted final dangles the last fini sender.
//! Uniqueness (no duplicate rows) is the sorted-keys argument here. `fini_val` and
//! `fini_epoch` are plain committed columns pinned by the bus match to FEXT_PAGE's
//! fini token (whose value is already canonical), exactly as `GLOBAL_MEMORY`'s
//! `FINI`/`FINI_EPOCH` are byte-checked transitively — do not add a redundant range
//! check on them here.
//!
//! ## Bus token (`GlobalFieldMemory`)
//!
//! `[domain, addr_lo, addr_hi, value, epoch]` — carries the domain (field-storage
//! spans 3/4/5, unlike RAM's domain-0-only `GlobalMemory`) and a full field-element
//! value. No timestamp: the cross-epoch chain is ordered by epoch.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::bitwise::BitwiseOperation;
use super::fext_sorted_keys::{self, SortedKeysLayout};
use super::local_to_global::GENESIS_EPOCH;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, zeroed_fe_vec};

/// Column layout for the shared `(domain, addr)` sorted-keys uniqueness argument.
const LAYOUT: SortedKeysLayout = SortedKeysLayout {
    domain: cols::DOMAIN,
    addr_0: cols::ADDR_0,
    addr_1: cols::ADDR_1,
    mu: cols::MU,
    addr0_hw_lo: cols::ADDR0_HW_LO,
    addr0_hw_hi: cols::ADDR0_HW_HI,
    addr1_hw_lo: cols::ADDR1_HW_LO,
    addr1_hw_hi: cols::ADDR1_HW_HI,
    next_addr_0: cols::NEXT_ADDR_0,
    next_addr_1: cols::NEXT_ADDR_1,
    same_dom: cols::SAME_DOM,
    sel_same: cols::SEL_SAME,
};

// =========================================================================
// Column indices
// =========================================================================

/// Column indices for the GLOBAL_FIELD_MEMORY table (one row per touched cell).
pub mod cols {
    /// Memory domain of this cell (3, 4, or 5).
    pub const DOMAIN: usize = 0;
    /// Cell address (DWordWL).
    pub const ADDR_0: usize = 1;
    pub const ADDR_1: usize = 2;
    /// Final value after the last touching epoch (pinned by the bus).
    pub const FINI_VAL: usize = 3;
    /// Last epoch that touched the cell (pinned by the bus).
    pub const FINI_EPOCH: usize = 4;
    /// Multiplicity bit / real-row selector.
    pub const MU: usize = 5;

    // --- uniqueness (sorted-keys) argument, mirroring FEXT_PAGE -------------
    /// Half-word decomposition of the two addr limbs, range-checked to `[0, 2^32)`
    /// via `IsHalfword` so the addr `<` ALU lookup is sound.
    pub const ADDR0_HW_LO: usize = 6;
    pub const ADDR0_HW_HI: usize = 7;
    pub const ADDR1_HW_LO: usize = 8;
    pub const ADDR1_HW_HI: usize = 9;
    /// The next row's addr limbs, copied in for the cross-row `addr < addr` compare.
    pub const NEXT_ADDR_0: usize = 10;
    pub const NEXT_ADDR_1: usize = 11;
    /// 1 iff this row and the next share a domain.
    pub const SAME_DOM: usize = 12;
    /// `μ_next · same_dom`: gates the addr strict-increase LT.
    pub const SEL_SAME: usize = 13;

    pub const NUM_COLUMNS: usize = 14;
}

// =========================================================================
// Types
// =========================================================================

/// One touched field-storage cell and its final state across the whole run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldCellFinal {
    pub domain: u64,
    pub addr: u64,
    /// Final value (canonical field element) after the last touching epoch.
    pub value: u64,
    /// Label of the last epoch that touched the cell.
    pub epoch: u64,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Build the GLOBAL_FIELD_MEMORY trace: one row per touched cell, sorted strictly
/// ascending by `(domain, addr)` with active rows contiguous at the top, padded to
/// a power of two (min 4). Padding rows carry a valid domain (3) and `μ = 0` so the
/// ungated domain constraint holds and they fire no bus interactions.
pub fn generate_global_field_trace(
    cells: &[FieldCellFinal],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut cells = cells.to_vec();
    cells.sort_by_key(|c| (c.domain, c.addr));

    let num_rows = cells.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, c) in cells.iter().enumerate() {
        table.set_fe(row, cols::DOMAIN, FE::from(c.domain));
        table.set_dword_wl(row, cols::ADDR_0, c.addr);
        table.set_fe(row, cols::FINI_VAL, FE::from(c.value));
        table.set_fe(row, cols::FINI_EPOCH, FE::from(c.epoch));
        table.set_fe(row, cols::MU, FE::one());
    }

    // Shared sorted-keys columns: addr half-words, padding domain, cross-row helpers.
    LAYOUT.fill_trace(table, cells.len(), num_rows);

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// Bus interactions on the cross-epoch `GlobalFieldMemory` bus (token
/// `[domain, addr_lo, addr_hi, value, epoch]`), plus the shared `(domain, addr)`
/// sorted-keys uniqueness lookups (addr-LT + addr-limb `IsHalfword`). Multiplicity
/// `MU`, so padding fires nothing.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = vec![
        // GFM-GENESIS: send the genesis token [domain, addr, 0, GENESIS]. Value and
        // epoch are constants — genesis field-storage is zero, ordered below every
        // real epoch — so neither is a prover-chosen column.
        BusInteraction::sender(
            BusId::GlobalFieldMemory,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                BusValue::constant(0),
                BusValue::constant(GENESIS_EPOCH),
            ],
        ),
        // GFM-FINAL: receive the finalization token [domain, addr, fini_val,
        // fini_epoch]. fini_val/fini_epoch are pinned by the match to FEXT_PAGE's
        // fini sender (whose value is already canonical) — no redundant check here.
        BusInteraction::receiver(
            BusId::GlobalFieldMemory,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                direct(cols::FINI_VAL),
                direct(cols::FINI_EPOCH),
            ],
        ),
    ];
    interactions.extend(LAYOUT.bus_interactions());
    interactions
}

/// The BITWISE `IsHalfword` rows the anchor's addr-limb range checks send (4 per
/// cell), which the global proof's BITWISE provider must count.
pub fn collect_bitwise(cells: &[FieldCellFinal]) -> Vec<BitwiseOperation> {
    fext_sorted_keys::collect_bitwise(cells.iter().map(|c| c.addr))
}

/// The addr `<` ALU LT ops the anchor's uniqueness sends (same-domain consecutive
/// cells), which the global proof's LT provider must receive.
pub fn collect_lt(cells: &[FieldCellFinal]) -> Vec<super::lt::LtOperation> {
    fext_sorted_keys::collect_lt(cells.iter().map(|c| (c.domain, c.addr)))
}

// =========================================================================
// Constraints
// =========================================================================

/// GLOBAL_FIELD_MEMORY constraints. Per-row: `IS_BIT(μ)` (0), domain `∈ {3,4,5}`
/// (1), `IS_BIT(same_dom)` (2), addr-limb recompose (3, 4). Transition (exempting
/// the last row): `μ` non-increasing (5), `sel_same` definition (6), same-domain ⇒
/// equal domain (7), domain increases by 1 or 2 on a change (8), next-addr copies
/// (9, 10). Mirrors FEXT_PAGE's sorted-keys uniqueness argument exactly.
pub struct GlobalFieldMemoryConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for GlobalFieldMemoryConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // The sparse anchor's entire constraint set is the shared sorted-keys
        // uniqueness argument (indices 0..=10); the genesis-value and finalization
        // bindings ride the GlobalFieldMemory bus, not extra AIR constraints.
        LAYOUT.emit_constraints(b);
    }

    fn max_degree(&self) -> usize {
        3
    }

    fn next_row_columns(&self) -> Vec<usize> {
        vec![cols::DOMAIN, cols::ADDR_0, cols::ADDR_1, cols::MU]
    }
}
