//! The `FIELD_VM_MEM` chip: read-only memory, one row per address.
//!
//! Not in the spec (see `FIELD_VM_SPEC_GAPS.md` §1). `addr` is pinned to the
//! row index (`addr₀ = 0`, `addr' = addr + 1`) so no address can be served with
//! two values.

use stark::constraints::boundary::BoundaryConstraint;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use stark::lookup::{BoundaryConstraintBuilder, BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::air::FvmPublicInputs;
use crate::tables::types::{BusId, FE, FEE, GoldilocksExtension, GoldilocksField, zeroed_fe_vec};

type F = GoldilocksField;
type E = GoldilocksExtension;

pub mod cols {
    pub const ADDR: usize = 0;
    pub const VALUE: usize = 1;
    pub const MU: usize = 4;
    pub const NUM_COLUMNS: usize = 5;
}

pub fn generate_trace(mem: &[FEE], mult: &[u64], min_rows: usize) -> TraceTable<F, E> {
    let rows = mem.len().next_power_of_two().max(min_rows);
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let t = &mut trace.main_table;
    for row in 0..rows {
        t.set(row, cols::ADDR, FE::from(row as u64));
    }
    for (row, (v, m)) in mem.iter().zip(mult).enumerate() {
        for (k, c) in v.value().iter().enumerate() {
            t.set(row, cols::VALUE + k, *c);
        }
        t.set(row, cols::MU, FE::from(*m));
    }
    trace
}

/// **Receives** `FIELD_VM_MEM[addr, 0, 0] -> value` with multiplicity `μ`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let direct = |c| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    vec![BusInteraction::receiver(
        BusId::FieldVmMem,
        Multiplicity::Column(cols::MU),
        vec![
            direct(cols::ADDR),
            BusValue::constant(0),
            BusValue::constant(0),
            direct(cols::VALUE),
            direct(cols::VALUE + 1),
            direct(cols::VALUE + 2),
        ],
    )]
}

#[derive(Clone, Copy, Default)]
pub struct MemConstraints;

impl ConstraintSet<F, E> for MemConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let e = b.main(1, cols::ADDR) - b.main(0, cols::ADDR) - b.one();
        b.emit_base_rows(0, RowDomain::except_last(1), e);
    }
}

pub struct MemBoundary;

impl BoundaryConstraintBuilder<F, E, FvmPublicInputs> for MemBoundary {
    fn boundary_constraints(_: &FvmPublicInputs, _: &[FEE]) -> Vec<BoundaryConstraint<E>> {
        vec![BoundaryConstraint::new_main(cols::ADDR, 0, FEE::zero())]
    }
}
