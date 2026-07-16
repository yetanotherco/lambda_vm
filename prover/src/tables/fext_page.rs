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
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::emit_is_bit;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op, zeroed_fe_vec,
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

        // Half-word range-check decomposition of the two 32-bit addr limbs.
        let lo = op.addr & 0xFFFF_FFFF;
        let hi = op.addr >> 32;
        table.set_fe(row, cols::ADDR0_HW_LO, FE::from(lo & 0xFFFF));
        table.set_fe(row, cols::ADDR0_HW_HI, FE::from(lo >> 16));
        table.set_fe(row, cols::ADDR1_HW_LO, FE::from(hi & 0xFFFF));
        table.set_fe(row, cols::ADDR1_HW_HI, FE::from(hi >> 16));
    }

    // Padding rows carry a valid domain (3) so the domain constraint holds; μ = 0
    // keeps them out of the bus.
    for row in ops.len()..num_rows {
        table.set_fe(row, cols::DOMAIN, FE::from(3u64));
    }

    // Cross-row helpers: copy the next row's addr, and set the same-domain flag
    // and LT selector. The last row's transition is exempt, so it keeps zeros.
    for row in 0..num_rows - 1 {
        let next_addr_0 = *table.get(row + 1, cols::ADDR_0);
        let next_addr_1 = *table.get(row + 1, cols::ADDR_1);
        let cur_dom = *table.get(row, cols::DOMAIN);
        let next_dom = *table.get(row + 1, cols::DOMAIN);
        let next_active = *table.get(row + 1, cols::MU) == FE::one();
        let same = cur_dom == next_dom;

        table.set_fe(row, cols::NEXT_ADDR_0, next_addr_0);
        table.set_fe(row, cols::NEXT_ADDR_1, next_addr_1);
        table.set_fe(
            row,
            cols::SAME_DOM,
            if same { FE::one() } else { FE::zero() },
        );
        table.set_fe(
            row,
            cols::SEL_SAME,
            if same && next_active {
                FE::one()
            } else {
                FE::zero()
            },
        );
    }

    trace
}

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// `IsHalfword[col]` — range-check that the column holds a valid half-word
/// `[0, 2^16)` (mult = μ).
fn is_halfword(col: usize) -> BusInteraction {
    BusInteraction::sender(
        BusId::IsHalfword,
        Multiplicity::Column(cols::MU),
        vec![direct(col)],
    )
}

/// Bus interactions: emit the zero-init token and consume the final token for
/// each touched cell, plus the uniqueness argument's `addr[i] < addr[i+1]` ALU
/// lookup and the addr-limb range checks.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
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
        // uniqueness: addr[i] < addr[i+1] on same-domain active transitions.
        // Sound because the addr limbs are pinned to `[0, 2^32)` half-words.
        BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(cols::SEL_SAME),
            vec![
                BusValue::Packed {
                    start_column: cols::ADDR_0,
                    packing: Packing::DWordWL,
                },
                BusValue::Packed {
                    start_column: cols::NEXT_ADDR_0,
                    packing: Packing::DWordWL,
                },
                BusValue::constant(alu_op::LT as u64),
                BusValue::constant(1),
                BusValue::constant(0),
            ],
        ),
        is_halfword(cols::ADDR0_HW_LO),
        is_halfword(cols::ADDR0_HW_HI),
        is_halfword(cols::ADDR1_HW_LO),
        is_halfword(cols::ADDR1_HW_HI),
    ]
}

/// FEXT_PAGE constraints. Per-row: `IS_BIT(μ)` (0), domain `∈ {3,4,5}` (1),
/// `IS_BIT(same_dom)` (2), addr-limb recompose (3, 4). Transition (exempting the
/// last row): `μ` non-increasing (5), `sel_same` definition (6), same-domain ⇒
/// equal domain (7), domain increases by 1 or 2 on a change (8), next-addr copies
/// (9, 10).
pub struct FextPageConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextPageConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);

        // Domain ∈ {3, 4, 5}: `(D - 3)(D - 4)(D - 5) = 0`. Ungated (degree 3,
        // within budget); padding rows carry domain 3 so it holds everywhere.
        let d = b.main(0, cols::DOMAIN);
        let three = b.const_base(3);
        let four = b.const_base(4);
        let five = b.const_base(5);
        b.emit_base(1, (d.clone() - three) * (d.clone() - four) * (d - five));

        emit_is_bit(b, 2, cols::SAME_DOM, None);

        // Addr-limb recompose: `ADDR_k = hw_lo + 2^16 * hw_hi`. With the
        // `IsHalfword` range checks this pins each limb to `[0, 2^32)`.
        let two16 = b.const_base(1 << 16);
        let a0 = b.main(0, cols::ADDR_0);
        let a0_lo = b.main(0, cols::ADDR0_HW_LO);
        let a0_hi = b.main(0, cols::ADDR0_HW_HI);
        b.emit_base(3, a0 - (a0_lo + two16.clone() * a0_hi));
        let a1 = b.main(0, cols::ADDR_1);
        let a1_lo = b.main(0, cols::ADDR1_HW_LO);
        let a1_hi = b.main(0, cols::ADDR1_HW_HI);
        b.emit_base(4, a1 - (a1_lo + two16.clone() * a1_hi));

        // --- transition constraints (read the next row) --------------------
        let tr = RowDomain::except_last(1);
        let one = b.one();
        let two = b.const_base(2);

        let mu_cur = b.main(0, cols::MU);
        let mu_next = b.main(1, cols::MU);
        let same = b.main(0, cols::SAME_DOM);
        let sel = b.main(0, cols::SEL_SAME);
        let d_cur = b.main(0, cols::DOMAIN);
        let d_next = b.main(1, cols::DOMAIN);

        // μ non-increasing: active rows are contiguous at the top.
        b.emit_base_rows(5, tr, mu_next.clone() * (one.clone() - mu_cur));

        // sel_same = μ_next · same_dom.
        b.emit_base_rows(6, tr, sel.clone() - mu_next.clone() * same);

        // same_dom (active) ⇒ equal domain.
        b.emit_base_rows(7, tr, sel.clone() * (d_next.clone() - d_cur.clone()));

        // ¬same_dom (active) ⇒ domain increases by 1 or 2. sel_diff = μ_next − sel.
        let sel_diff = mu_next - sel;
        let delta = d_next - d_cur;
        b.emit_base_rows(8, tr, sel_diff * (delta.clone() - one) * (delta - two));

        // next_addr copies feed the cross-row LT.
        let na0 = b.main(0, cols::NEXT_ADDR_0);
        let addr0_next = b.main(1, cols::ADDR_0);
        b.emit_base_rows(9, tr, na0 - addr0_next);
        let na1 = b.main(0, cols::NEXT_ADDR_1);
        let addr1_next = b.main(1, cols::ADDR_1);
        b.emit_base_rows(10, tr, na1 - addr1_next);

        // sel_same is a bit on EVERY row, including the last (whose definition,
        // idx 6, is exempt). Without this a prover could set the addr-LT
        // multiplicity to -1 on the last row and cancel an invalid `addr < next`
        // lookup elsewhere, re-introducing duplicate (domain, addr) keys.
        emit_is_bit(b, 11, cols::SEL_SAME, None);
    }

    fn max_degree(&self) -> usize {
        3
    }
}
