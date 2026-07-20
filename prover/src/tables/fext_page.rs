//! FEXT_PAGE table: init/finalization bookend for the field-storage memory
//! domains (3/4/5), analogous to `PAGE` (RAM, domain 0) and `REGISTER`
//! (domain 1) but for full field-element values.
//!
//! One row per field-storage cell `(domain, addr)` touched by any FEXT op. It
//! emits the cell's zero-init token and consumes its final token, closing the
//! `Memory`-bus chain the FEXT_LOAD/FEXT_FMA accesses open:
//! - **Receiver** on `Memory`: `[domain, addr, 0, 0]` — emits the zero init token
//!   (balances the first access's consume-old).
//! - **Sender** on `Memory`: `[domain, addr, final_ts, final_val]` — consumes the
//!   final token (balances the last access's emit-new).
//!
//! Field-storage is zero-initialized (scratch, single-proof scope), so `init` is
//! the constant 0 rather than a committed column.
//!
//! ## Soundness: domain and uniqueness
//! The domain and address feed the shared `Memory` bus, so they must be pinned:
//! - **Domain** is constrained to `{3, 4, 5}` (idx 1), otherwise a prover could
//!   forge tokens in another domain's chain (e.g. domain 0 = RAM).
//! - **Uniqueness** of each active `(domain, addr)` is enforced by a sorted-keys
//!   argument: rows are emitted sorted strictly ascending by `(domain, addr)`,
//!   with active rows contiguous at the top. Two rows for the same cell would
//!   emit two init tokens `[domain, addr, 0, 0]`, letting a prover reset a cell
//!   to zero mid-execution. The strict-increase constraints (idx 5..=10, plus the
//!   addr `<` ALU lookup) make the keys distinct.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::fext_sorted_keys::SortedKeysLayout;
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

/// Column indices for the FEXT_PAGE table.
pub mod cols {
    /// Memory domain of this cell (3, 4, or 5).
    pub const DOMAIN: usize = 0;
    /// Cell address (DWordWL).
    pub const ADDR_0: usize = 1;
    pub const ADDR_1: usize = 2;
    /// Timestamp of the last access to this cell (DWordWL).
    pub const FINAL_TS_0: usize = 3;
    pub const FINAL_TS_1: usize = 4;
    /// Final value stored in this cell.
    pub const FINAL_VAL: usize = 5;
    /// Multiplicity bit.
    pub const MU: usize = 6;

    // --- uniqueness (sorted-keys) argument ---------------------------------
    /// Half-word decomposition of the two addr limbs, range-checking each to
    /// `[0, 2^32)` via `IsHalfword` so the addr `<` ALU lookup is sound (the LT
    /// chip assumes word-sized limbs).
    pub const ADDR0_HW_LO: usize = 7;
    pub const ADDR0_HW_HI: usize = 8;
    pub const ADDR1_HW_LO: usize = 9;
    pub const ADDR1_HW_HI: usize = 10;
    /// The next row's addr limbs, copied in so the current-row-only bus can run
    /// the cross-row `addr[i] < addr[i+1]` comparison.
    pub const NEXT_ADDR_0: usize = 11;
    pub const NEXT_ADDR_1: usize = 12;
    /// 1 iff this row and the next share a domain.
    pub const SAME_DOM: usize = 13;
    /// `μ_next · same_dom`: gates the addr strict-increase LT (materialized
    /// because multiplicities cannot be products).
    pub const SEL_SAME: usize = 14;

    pub const NUM_COLUMNS: usize = 15;
}

/// One touched field-storage cell and its final state.
#[derive(Debug, Clone)]
pub struct FextPageOperation {
    pub domain: u64,
    pub addr: u64,
    pub final_ts: u64,
    pub final_val: u64,
}

/// Generates the FEXT_PAGE trace (one row per touched cell, padded to next power
/// of two, min 4). Rows are sorted strictly ascending by `(domain, addr)` with
/// active rows contiguous at the top; padding rows are `μ = 0` and carry a valid
/// domain (3) so the ungated domain constraint holds everywhere.
pub fn generate_fext_page_trace(
    ops: &[FextPageOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut ops = ops.to_vec();
    ops.sort_by_key(|o| (o.domain, o.addr));

    let num_rows = ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, op) in ops.iter().enumerate() {
        table.set_fe(row, cols::DOMAIN, FE::from(op.domain));
        table.set_dword_wl(row, cols::ADDR_0, op.addr);
        table.set_dword_wl(row, cols::FINAL_TS_0, op.final_ts);
        table.set_fe(row, cols::FINAL_VAL, FE::from(op.final_val));
        table.set_fe(row, cols::MU, FE::one());
    }

    // Shared sorted-keys columns: addr half-words, padding domain, cross-row helpers.
    LAYOUT.fill_trace(table, ops.len(), num_rows);

    trace
}

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// Bus interactions: emit the zero-init token and consume the final token for
/// each touched cell on the `Memory` bus, plus the shared `(domain, addr)`
/// sorted-keys uniqueness lookups (addr-LT + addr-limb `IsHalfword`).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = vec![
        // init: emit [domain, addr, ts=0, value=0]
        BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
        // fini: consume [domain, addr, final_ts, final_val]
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Column(cols::MU),
            vec![
                direct(cols::DOMAIN),
                direct(cols::ADDR_0),
                direct(cols::ADDR_1),
                direct(cols::FINAL_TS_0),
                direct(cols::FINAL_TS_1),
                direct(cols::FINAL_VAL),
            ],
        ),
    ];
    interactions.extend(LAYOUT.bus_interactions());
    interactions
}

/// FEXT_PAGE constraints. Per-row: `IS_BIT(μ)` (0), domain `∈ {3,4,5}` (1),
/// `IS_BIT(same_dom)` (2), addr-limb recompose (3, 4). Transition (exempting the
/// last row): `μ` non-increasing (5), `sel_same` definition (6), same-domain ⇒
/// equal domain (7), domain increases by 1 or 2 on a change (8), next-addr copies
/// (9, 10). Per-row again: `IS_BIT(sel_same)` (11), pinning the LT sender's
/// multiplicity to `{0,1}` on the last row too.
pub struct FextPageConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextPageConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // FEXT_PAGE's entire constraint set is the shared sorted-keys uniqueness
        // argument (IS_BIT(μ), domain ∈ {3,4,5}, addr recompose, strict-ascending
        // transitions, IS_BIT(sel_same)), indices 0..=11.
        LAYOUT.emit_constraints(b);
    }

    fn max_degree(&self) -> usize {
        3
    }

    fn next_row_columns(&self) -> Vec<usize> {
        // Constraints 5-10 read the next row via `main(1, ·)`: the contiguity and
        // domain-ordering checks (DOMAIN, ADDR_0, ADDR_1) and μ non-increasing (MU).
        vec![cols::DOMAIN, cols::ADDR_0, cols::ADDR_1, cols::MU]
    }
}
