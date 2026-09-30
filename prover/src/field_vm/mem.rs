//! The `FIELD_VM_MEM` chip: read-only memory, one row per address.
//!
//! Not in the spec (see `FIELD_VM_SPEC_GAPS.md` §1). `addr` is pinned to the
//! row index (`addr₀ = 0`, `addr' = addr + 1`) so no address can be served with
//! two values.
//!
//! Cells flagged `pub` are also sent on `FIELD_VM_PUBLIC`, which the verifier
//! closes against the `(addr, value)` pairs it is given. Addresses being
//! unique, the flagged cells are then exactly the claimed ones, whatever their
//! position: that binds the inputs the program reads and the outputs it leaves.

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
    pub const PUB: usize = 5;
    pub const NUM_COLUMNS: usize = 6;
    /// With the LFM hash chips: the fourth lane of a cell they write, and how
    /// many times MEM sends (`M_OUT`) and receives (`M_IN`) the cell on `LfmMem`.
    pub const V3: usize = 6;
    pub const M_OUT: usize = 7;
    pub const M_IN: usize = 8;
    pub const NUM_COLUMNS_LFM: usize = 9;
}

/// A cell MEM shares with the LFM hash chips: `(addr, sends, receives, lane 3)`.
pub type LfmCell = (u64, u64, u64, FE);

/// [`generate_trace`] with the `LfmMem` columns.
pub fn generate_trace_lfm(
    mem: &[FEE],
    mult: &[u64],
    public: &[u64],
    min_rows: usize,
    lfm: &[LfmCell],
) -> TraceTable<F, E> {
    let base = generate_trace(mem, mult, public, min_rows);
    let rows = base.num_rows();
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(rows * cols::NUM_COLUMNS_LFM),
        cols::NUM_COLUMNS_LFM,
        1,
    );
    let t = &mut trace.main_table;
    for r in 0..rows {
        for c in 0..cols::NUM_COLUMNS {
            t.set(r, c, *base.main_table.get(r, c));
        }
    }
    for &(addr, out, inn, v3) in lfm {
        let r = addr as usize;
        t.set(r, cols::M_OUT, FE::from(out));
        t.set(r, cols::M_IN, FE::from(inn));
        t.set(r, cols::V3, v3);
    }
    trace
}

/// [`bus_interactions`] plus the `LfmMem` sender `(addr, value, 0)` and
/// receiver `(addr, value, v3)`. Addresses are unique, so the witness
/// multiplicities can only match the hash chips' fixed reads and writes.
pub fn bus_interactions_lfm() -> Vec<BusInteraction> {
    let direct = |c| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    let mut v = bus_interactions();
    let token = |lane3: BusValue| {
        vec![
            direct(cols::ADDR),
            direct(cols::VALUE),
            direct(cols::VALUE + 1),
            direct(cols::VALUE + 2),
            lane3,
        ]
    };
    v.push(BusInteraction::sender(
        BusId::LfmMem,
        Multiplicity::Column(cols::M_OUT),
        token(BusValue::constant(0)),
    ));
    v.push(BusInteraction::receiver(
        BusId::LfmMem,
        Multiplicity::Column(cols::M_IN),
        token(direct(cols::V3)),
    ));
    v
}

/// `public` are the addresses whose cells are flagged; each must be `< mem.len()`.
pub fn generate_trace(
    mem: &[FEE],
    mult: &[u64],
    public: &[u64],
    min_rows: usize,
) -> TraceTable<F, E> {
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
    for &addr in public {
        assert!(
            (addr as usize) < mem.len(),
            "public cell {addr} outside memory"
        );
        t.set(addr as usize, cols::PUB, FE::one());
    }
    trace
}

/// - **Receives** `FIELD_VM_MEM[addr, 0, 0] -> value` with multiplicity `μ`.
/// - **Sends** `FIELD_VM_PUBLIC[addr, value]` when `pub`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let direct = |c| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    vec![
        BusInteraction::receiver(
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
        ),
        BusInteraction::sender(
            BusId::FieldVmPublic,
            Multiplicity::Column(cols::PUB),
            vec![
                direct(cols::ADDR),
                direct(cols::VALUE),
                direct(cols::VALUE + 1),
                direct(cols::VALUE + 2),
            ],
        ),
    ]
}

/// `lfm`: a row either sends or receives on `LfmMem`, so MEM cannot relay a
/// hash-chip cell with its fourth lane zeroed.
#[derive(Clone, Copy, Default)]
pub struct MemConstraints {
    pub lfm: bool,
}

impl ConstraintSet<F, E> for MemConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let e = b.main(1, cols::ADDR) - b.main(0, cols::ADDR) - b.one();
        b.emit_base_rows(0, RowDomain::except_last(1), e);
        let p = b.main(0, cols::PUB);
        b.emit_base(1, p.clone() * (p - b.one()));
        if self.lfm {
            let both = b.main(0, cols::M_OUT) * b.main(0, cols::M_IN);
            b.emit_base(2, both);
        }
    }
}

pub struct MemBoundary;

impl BoundaryConstraintBuilder<F, E, FvmPublicInputs> for MemBoundary {
    fn boundary_constraints(_: &FvmPublicInputs, _: &[FEE]) -> Vec<BoundaryConstraint<E>> {
        vec![BoundaryConstraint::new_main(cols::ADDR, 0, FEE::zero())]
    }
}
