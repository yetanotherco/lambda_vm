//! The `FIELD_VM_BRIDGE` chip: the cells that cross between the Field VM's
//! memory and the LFM hash-side chips' `LfmMem` bus.
//!
//! One preprocessed row per crossing cell. Every row reads the cell from MEM
//! (`FIELD_VM_MEM` sender, like any Field VM read); an outgoing row then
//! sends it on `LfmMem` with the program's read count, an incoming row
//! receives the hash side's write of it. MEM holds three lanes, `LfmMem`
//! four: the fourth is a witness column, pinned to the preprocessed `L3C`
//! unless the row is a word (a hint, or an incoming cell).

use stark::config::Commitment;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;

use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, zeroed_fe_vec};

type F = GoldilocksField;
type E = GoldilocksExtension;

pub mod cols {
    /// The cell's Field VM memory address.
    pub const FADDR: usize = 0;
    /// The cell's LFM address.
    pub const LADDR: usize = 1;
    pub const MULT_OUT: usize = 2;
    pub const IS_IN: usize = 3;
    pub const IS_WORD: usize = 4;
    pub const L3C: usize = 5;
    pub const IS_REAL: usize = 6;
    pub const NUM_PRECOMPUTED_COLS: usize = 7;
    pub const V0: usize = NUM_PRECOMPUTED_COLS;
    pub const NUM_COLUMNS: usize = V0 + 4;
}

#[derive(Clone, Debug)]
pub struct BridgeRow {
    pub faddr: u64,
    pub laddr: u64,
    /// Hash-side reads of an outgoing cell; zero for an incoming one.
    pub mult_out: u64,
    pub is_in: bool,
    pub is_word: bool,
    /// Lane 3 of a non-word outgoing cell.
    pub l3c: FE,
}

fn fill_prep(rows: &[BridgeRow], min_rows: usize) -> TraceTable<F, E> {
    let height = rows.len().next_power_of_two().max(min_rows);
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(height * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let t = &mut trace.main_table;
    for (i, r) in rows.iter().enumerate() {
        t.set(i, cols::FADDR, FE::from(r.faddr));
        t.set(i, cols::LADDR, FE::from(r.laddr));
        t.set(i, cols::MULT_OUT, FE::from(r.mult_out));
        t.set(i, cols::IS_IN, FE::from(r.is_in as u64));
        t.set(i, cols::IS_WORD, FE::from(r.is_word as u64));
        t.set(i, cols::L3C, r.l3c);
        t.set(i, cols::IS_REAL, FE::one());
    }
    trace
}

/// `value(i)` is the four-lane word of row `i`.
pub fn generate_trace(
    rows: &[BridgeRow],
    value: impl Fn(usize) -> [FE; 4],
    min_rows: usize,
) -> TraceTable<F, E> {
    let mut trace = fill_prep(rows, min_rows);
    let t = &mut trace.main_table;
    for i in 0..rows.len() {
        for (k, v) in value(i).into_iter().enumerate() {
            t.set(i, cols::V0 + k, v);
        }
    }
    trace
}

/// The Merkle root of the preprocessed columns.
pub fn commitment(rows: &[BridgeRow], options: &ProofOptions, min_rows: usize) -> Commitment {
    let trace = fill_prep(rows, min_rows);
    let columns: Vec<Vec<FE>> = (0..cols::NUM_PRECOMPUTED_COLS)
        .map(|c| {
            (0..trace.num_rows())
                .map(|r| *trace.main_table.get(r, c))
                .collect()
        })
        .collect();
    crate::lfm::commit::commit_columns(&columns, options)
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let direct = |c| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    let word = || {
        let mut v = vec![direct(cols::LADDR)];
        v.extend((0..4).map(|k| direct(cols::V0 + k)));
        v
    };
    vec![
        BusInteraction::sender(
            BusId::FieldVmMem,
            Multiplicity::Column(cols::IS_REAL),
            vec![
                direct(cols::FADDR),
                BusValue::constant(0),
                BusValue::constant(0),
                direct(cols::V0),
                direct(cols::V0 + 1),
                direct(cols::V0 + 2),
            ],
        ),
        BusInteraction::sender(BusId::LfmMem, Multiplicity::Column(cols::MULT_OUT), word()),
        BusInteraction::receiver(BusId::LfmMem, Multiplicity::Column(cols::IS_IN), word()),
    ]
}

#[derive(Clone, Copy, Default)]
pub struct BridgeConstraints;

impl ConstraintSet<F, E> for BridgeConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let free = b.one() - b.main(0, cols::IS_WORD);
        let lane3 = b.main(0, cols::V0 + 3) - b.main(0, cols::L3C);
        b.emit_base(0, free * lane3);
    }
}
