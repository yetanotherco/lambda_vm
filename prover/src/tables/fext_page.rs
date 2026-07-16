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
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::emit_is_bit;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, zeroed_fe_vec};

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

    pub const NUM_COLUMNS: usize = 7;
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
/// of two, min 4). Padding rows are all-zero (`μ = 0`) and contribute nothing.
pub fn generate_fext_page_trace(
    ops: &[FextPageOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
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

    // Padding rows carry a valid domain (3) so the ungated domain constraint
    // holds on every row; μ = 0 keeps them out of the bus.
    for row in ops.len()..num_rows {
        table.set_fe(row, cols::DOMAIN, FE::from(3u64));
    }

    trace
}

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// Bus interactions: emit the zero-init token and consume the final token for
/// each touched cell.
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
    ]
}

/// FEXT_PAGE constraints: idx 0 is `IS_BIT(μ)`, idx 1 pins the domain to
/// `{3, 4, 5}` (the field-storage coefficient domains).
pub struct FextPageConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for FextPageConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);

        // Domain ∈ {3, 4, 5}: `(D - 3)(D - 4)(D - 5) = 0`. The domain feeds the
        // shared Memory bus, so leaving it a free witness would let a prover forge
        // tokens in another domain's chain (e.g. domain 0 = RAM). Ungated (degree
        // 3, within budget); padding rows carry domain 3 so it holds everywhere.
        let d = b.main(0, cols::DOMAIN);
        let three = b.const_base(3);
        let four = b.const_base(4);
        let five = b.const_base(5);
        b.emit_base(1, (d.clone() - three) * (d.clone() - four) * (d - five));
    }

    fn max_degree(&self) -> usize {
        3
    }
}
