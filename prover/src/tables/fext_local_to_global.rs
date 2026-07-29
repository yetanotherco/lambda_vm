//! FEXT_LOCAL_TO_GLOBAL table: per-epoch field-storage boundary claims for
//! cross-epoch continuations (the field-storage analog of `local_to_global.rs`).
//!
//! Under continuation, field-storage (memory domains 3/4/5) is carried across
//! epochs. This table replaces FEXT_PAGE's monolithic zero-init bookend for
//! continuation epochs: for every field cell `(domain, addr)` an epoch touches it
//!
//! - **bookends the epoch-local `Memory` bus** — receives the cell's carried init
//!   token `[domain, addr, ts=0, init_val]` (balancing the first FEXT access's
//!   consume-old) and sends its final token `[domain, addr, final_ts, final_val]`
//!   (balancing the last access's emit-new);
//! - **emits cross-epoch `GlobalFieldMemory` tokens** — receives the init token
//!   `[domain, addr, init_val, init_epoch]` left by the epoch that last wrote the
//!   cell and sends the fini token `[domain, addr, final_val, epoch_label]` for the
//!   next epoch, matched across epochs by the GLOBAL_FIELD_MEMORY aggregation.
//!
//! Mirrors `local_to_global.rs` exactly, with three field-storage differences:
//! - the token carries the domain (3/4/5) and a full field-element value, not a byte;
//! - **uniqueness of `(domain, addr)` is explicit** (the sorted-keys argument, as in
//!   FEXT_PAGE), because — unlike RAM, whose L2G leans on the MEMW genesis token —
//!   the FEXT accesses emit their own consume-old tokens and nothing external pins
//!   one init per cell (see `fext_page.rs`);
//! - `init_val` is a single committed column, pinned canonical by the cross-epoch
//!   bus (it must match either the anchor's genesis 0 or a prior epoch's canonical
//!   `final_val`), so it needs no explicit `< p` check — exactly as FEXT_PAGE's
//!   `final_val` and RAM's GLOBAL_MEMORY `FINI` are pinned transitively.
//!
//! `init_epoch` is the only cross-epoch-only quantity with no bus partner, so it is
//! range-checked here (two `IsHalfword` halfwords) and ordered by
//! `IsB20[epoch_label − 1 − init_epoch]`, forcing `init_epoch < fini_epoch`.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
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

/// Column indices for the FEXT_LOCAL_TO_GLOBAL table (one row per touched cell).
pub mod cols {
    /// Memory domain of this cell (3, 4, or 5).
    pub const DOMAIN: usize = 0;
    /// Cell address (DWordWL).
    pub const ADDR_0: usize = 1;
    pub const ADDR_1: usize = 2;
    /// Carried init value (pinned canonical by the GlobalFieldMemory bus).
    pub const INIT_VAL: usize = 3;
    /// Originating epoch as two `IsHalfword`-checked halfwords (GlobalFieldMemory-only).
    pub const INIT_EPOCH_0: usize = 4;
    pub const INIT_EPOCH_1: usize = 5;
    /// Timestamp of the last access to this cell this epoch (DWordWL).
    pub const FINAL_TS_0: usize = 6;
    pub const FINAL_TS_1: usize = 7;
    /// Final value at this epoch's end.
    pub const FINAL_VAL: usize = 8;
    /// Multiplicity bit / real-row selector.
    pub const MU: usize = 9;

    // --- uniqueness (sorted-keys) argument, mirroring FEXT_PAGE -------------
    pub const ADDR0_HW_LO: usize = 10;
    pub const ADDR0_HW_HI: usize = 11;
    pub const ADDR1_HW_LO: usize = 12;
    pub const ADDR1_HW_HI: usize = 13;
    pub const NEXT_ADDR_0: usize = 14;
    pub const NEXT_ADDR_1: usize = 15;
    pub const SAME_DOM: usize = 16;
    pub const SEL_SAME: usize = 17;

    pub const NUM_COLUMNS: usize = 18;

    /// Cross-epoch-only halfword columns, in order — every `IsHalfword`-checked
    /// column that has no `Memory`-bus partner.
    pub const RANGE_CHECKED_HALFWORDS: [usize; 2] = [INIT_EPOCH_0, INIT_EPOCH_1];
    /// Address-limb halfwords, `IsHalfword`-checked so the uniqueness LT is sound.
    pub const ADDR_HALFWORDS: [usize; 4] = [ADDR0_HW_LO, ADDR0_HW_HI, ADDR1_HW_LO, ADDR1_HW_HI];
}

// =========================================================================
// Types
// =========================================================================

/// The init/fini boundary claim for one touched field cell in one epoch.
/// Prover-local only (holds cell values); never serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldCellBoundary {
    pub domain: u64,
    pub addr: u64,
    /// Value the cell held when this epoch first touched it.
    pub init_val: u64,
    /// Epoch that last wrote the cell (or `GENESIS_EPOCH`).
    pub init_epoch: u64,
    /// Value the cell holds at this epoch's end.
    pub final_val: u64,
    /// Last access timestamp for the cell this epoch.
    pub final_ts: u64,
}

/// One epoch's touched field cells, each as `(domain, addr, final_value, final_ts)`
/// — the field-storage analog of [`super::local_to_global::EpochTouches`], produced
/// by the trace builder and turned into [`FieldCellBoundary`]s by the driver.
pub type FieldTouches = Vec<(u64, u64, u64, u64)>;

/// Per-cell field-storage provenance: `(domain, addr) → (last_writer_epoch, value)`.
/// Unset cells read back as the genesis default `(GENESIS_EPOCH, 0)`. No timestamp:
/// the cross-epoch init token is seeded at ts=0 (timestamps are epoch-local).
pub type FieldProvenance = std::collections::HashMap<(u64, u64), (u64, u64)>;

/// One epoch's field boundaries: take each touched cell's `init` from the running
/// `provenance` (its last writer + value) and record this epoch (1-based `epoch`
/// label) as the new writer of its `final` value. Mirrors
/// [`super::local_to_global::epoch_boundary`] for field-storage cells.
pub fn field_epoch_boundary(
    provenance: &mut FieldProvenance,
    epoch: u64,
    touched: &FieldTouches,
) -> Vec<FieldCellBoundary> {
    let mut boundaries = Vec::with_capacity(touched.len());
    for &(domain, addr, final_val, final_ts) in touched {
        let (init_epoch, init_val) = provenance
            .get(&(domain, addr))
            .copied()
            .unwrap_or((GENESIS_EPOCH, 0));
        boundaries.push(FieldCellBoundary {
            domain,
            addr,
            init_val,
            init_epoch,
            final_val,
            final_ts,
        });
        provenance.insert((domain, addr), (epoch, final_val));
    }
    boundaries
}

// =========================================================================
// Trace generation
// =========================================================================

/// Half-words of an epoch label (genesis 0 or a small 1-based index).
fn epoch_halfwords(epoch: u64) -> [u64; 2] {
    debug_assert!(epoch < (1 << 32), "epoch label exceeds 32 bits");
    [epoch & 0xFFFF, (epoch >> 16) & 0xFFFF]
}

/// Build the FEXT_LOCAL_TO_GLOBAL trace: one row per touched cell, sorted strictly
/// ascending by `(domain, addr)` with active rows contiguous at the top, padded to
/// a power of two (min 4). Padding rows carry a valid domain (3) and `μ = 0`.
pub fn generate_fext_local_to_global_trace(
    boundaries: &[FieldCellBoundary],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut boundaries = boundaries.to_vec();
    boundaries.sort_by_key(|b| (b.domain, b.addr));

    let num_rows = boundaries.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, b) in boundaries.iter().enumerate() {
        let init_epoch = epoch_halfwords(b.init_epoch);
        table.set_fe(row, cols::DOMAIN, FE::from(b.domain));
        table.set_dword_wl(row, cols::ADDR_0, b.addr);
        table.set_fe(row, cols::INIT_VAL, FE::from(b.init_val));
        table.set_fe(row, cols::INIT_EPOCH_0, FE::from(init_epoch[0]));
        table.set_fe(row, cols::INIT_EPOCH_1, FE::from(init_epoch[1]));
        table.set_dword_wl(row, cols::FINAL_TS_0, b.final_ts);
        table.set_fe(row, cols::FINAL_VAL, FE::from(b.final_val));
        table.set_fe(row, cols::MU, FE::one());
    }

    // Shared sorted-keys columns: addr half-words, padding domain, cross-row helpers.
    LAYOUT.fill_trace(table, boundaries.len(), num_rows);

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

/// A 32-bit value reconstructed from its two halfword columns: `lo + 2^16·hi`.
fn word(lo_col: usize, hi_col: usize) -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: lo_col,
        },
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: hi_col,
        },
    ])
}

fn mu() -> Multiplicity {
    Multiplicity::Column(cols::MU)
}

/// Cross-epoch `GlobalFieldMemory` bus interactions (two per touched cell):
/// receive the init token `[domain, addr, init_val, init_epoch]` left by the
/// originating epoch, and send the fini token `[domain, addr, final_val,
/// epoch_label]` for the next epoch. Matched ACROSS epochs by the GLOBAL_FIELD_MEMORY
/// aggregation, so within one epoch's table this bus is deliberately unbalanced.
pub fn global_bus_interactions(epoch_label: u64) -> Vec<BusInteraction> {
    vec![
        BusInteraction::receiver(
            BusId::GlobalFieldMemory,
            mu(),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                direct(cols::INIT_VAL),
                word(cols::INIT_EPOCH_0, cols::INIT_EPOCH_1),
            ],
        ),
        BusInteraction::sender(
            BusId::GlobalFieldMemory,
            mu(),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                direct(cols::FINAL_VAL),
                BusValue::constant(epoch_label),
            ],
        ),
    ]
}

/// Epoch-local `Memory` bus bookend (token `[domain, addr_lo, addr_hi, ts_lo,
/// ts_hi, value]`), replacing FEXT_PAGE's init/fini for touched cells: receive the
/// carried init token at ts=0, send the final token at the last access timestamp.
pub fn memory_bus_interactions() -> Vec<BusInteraction> {
    vec![
        BusInteraction::receiver(
            BusId::Memory,
            mu(),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                BusValue::constant(0),
                BusValue::constant(0),
                direct(cols::INIT_VAL),
            ],
        ),
        BusInteraction::sender(
            BusId::Memory,
            mu(),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                direct(cols::FINAL_TS_0),
                direct(cols::FINAL_TS_1),
                direct(cols::FINAL_VAL),
            ],
        ),
    ]
}

/// Range-check, ordering and uniqueness bus interactions (all with the right
/// multiplicity so padding fires none):
/// - `IsHalfword` per addr-limb halfword and per `init_epoch` halfword;
/// - `IsB20[epoch_label − 1 − init_epoch]`, forcing `init_epoch < fini_epoch`;
/// - the uniqueness `addr[i] < addr[i+1]` ALU LT on same-domain transitions.
pub fn range_check_interactions(epoch_label: u64) -> Vec<BusInteraction> {
    debug_assert!(epoch_label >= 1, "epoch_label must be a 1-based fini epoch");
    // Shared addr-LT + addr-limb IsHalfword, then the cross-epoch-only init_epoch
    // IsHalfword + the IsB20 ordering check (init_epoch < fini_epoch).
    let mut interactions = LAYOUT.bus_interactions();

    for &column in &cols::RANGE_CHECKED_HALFWORDS {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu(),
            vec![direct(column)],
        ));
    }

    interactions.push(BusInteraction::sender(
        BusId::IsB20,
        mu(),
        vec![BusValue::linear(vec![
            LinearTerm::Constant(epoch_label as i64 - 1),
            LinearTerm::Column {
                coefficient: -1,
                column: cols::INIT_EPOCH_0,
            },
            LinearTerm::Column {
                coefficient: -(1 << 16),
                column: cols::INIT_EPOCH_1,
            },
        ])],
    ));

    interactions
}

/// The BITWISE lookups the range checks send, so the BITWISE table's multiplicities
/// balance [`range_check_interactions`]' `IsHalfword` and `IsB20` senders. Padding
/// rows (`MU = 0`) fire nothing, so none are emitted for them. Keep in sync with
/// [`range_check_interactions`].
pub fn collect_bitwise_from_fext_l2g(
    boundaries: &[FieldCellBoundary],
    epoch_label: u64,
) -> Vec<BitwiseOperation> {
    // Shared addr-limb halfwords (4/cell), then the cross-epoch-only init_epoch
    // halfwords (2/cell) + IsB20 ordering (1/cell). BITWISE is a histogram, so order
    // does not matter.
    let mut ops = fext_sorted_keys::collect_bitwise(boundaries.iter().map(|b| b.addr));
    for b in boundaries {
        for v16 in epoch_halfwords(b.init_epoch) {
            ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (v16 & 0xFF) as u8,
                ((v16 >> 8) & 0xFF) as u8,
            ));
        }
        // Ordering: IsB20[epoch_label - 1 - init_epoch] (init_epoch < fini_epoch).
        let diff = epoch_label - 1 - b.init_epoch;
        ops.push(BitwiseOperation::b20(
            (diff & 0xFF) as u8,
            ((diff >> 8) & 0xFF) as u8,
            ((diff >> 16) & 0xF) as u8,
        ));
    }
    ops
}

/// The addr `<` ALU LT ops the uniqueness argument needs (same-domain consecutive
/// touched cells), which the epoch's LT table must receive. Derived from the sorted
/// touched-cell set (available at trace-build time), so it does not depend on the
/// driver-computed init_epoch.
pub fn collect_lt_from_touches(touched: &FieldTouches) -> Vec<super::lt::LtOperation> {
    fext_sorted_keys::collect_lt(touched.iter().map(|&(domain, addr, _, _)| (domain, addr)))
}

// =========================================================================
// Constraints
// =========================================================================

/// FEXT_LOCAL_TO_GLOBAL constraints: `IS_BIT(μ)` (0), domain `∈ {3,4,5}` (1),
/// `IS_BIT(same_dom)` (2), addr-limb recompose (3, 4), the sorted-keys uniqueness
/// transition constraints (5..=10), and `IS_BIT(sel_same)` (11), identical in shape
/// to FEXT_PAGE.
pub struct FextLocalToGlobalConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextLocalToGlobalConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // The shared sorted-keys uniqueness argument (indices 0..=11); the cross-epoch
        // ordering (IsB20) and value bindings ride the buses, not extra AIR constraints.
        LAYOUT.emit_constraints(b);
    }

    fn max_degree(&self) -> usize {
        3
    }

    fn next_row_columns(&self) -> Vec<usize> {
        vec![cols::DOMAIN, cols::ADDR_0, cols::ADDR_1, cols::MU]
    }
}
