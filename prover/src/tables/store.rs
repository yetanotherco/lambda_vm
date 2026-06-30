//! STORE table.
//!
//! Receives the high-level `MEMORY` op from the CPU for store instructions and
//! emits the low-level `MEMW` write. Spec: `spec/src/store.toml`.
//!
//! ## `memory_op` flag bit (spec-faithful)
//! The `MEMORY` receiver flags are `1 + 4·write2 + 8·write4 + 16·write8`; the
//! `+1` is `memory_op`, which balances against the CPU's `mem_flags`
//! (`memory_op = 1` for stores). This matches `store.toml`.
//!
//! Note: the `MEMW` *write* fingerprint carries no `old` value — the
//! previous memory contents are handled inside the MEMW table. So STORE needs
//! no `old` column (the MEMW *write* fingerprint omits `old`).
//!
//! ## Columns
//! - `base_address`: DWordWL (2 words) — effective write address
//! - `timestamp`: DWordWL (2 words)
//! - `write2`/`write4`/`write8`: Bit — exclusive width flags (1 byte = none set)
//! - `value`: DWordBL (8 bytes) — value to store
//! - `μ`: multiplicity

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraint_ir::{Capture, IrBuilder};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::new_is_bit_constraints;

// =========================================================================
// Column indices for STORE table
// =========================================================================

/// Column definitions for the STORE table.
pub mod cols {
    pub const BASE_ADDRESS_0: usize = 0;
    pub const BASE_ADDRESS_1: usize = 1;
    pub const TIMESTAMP_0: usize = 2;
    pub const TIMESTAMP_1: usize = 3;
    pub const WRITE2: usize = 4;
    pub const WRITE4: usize = 5;
    pub const WRITE8: usize = 6;
    /// value as 8 bytes (DWordBL), little-endian.
    pub const VALUE: [usize; 8] = [7, 8, 9, 10, 11, 12, 13, 14];
    /// μ: multiplicity
    pub const MU: usize = 15;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 16;
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single STORE operation. Exactly one of `write2/write4/write8` is set, or
/// none for a single-byte store.
#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct StoreOperation {
    pub base_address: u64,
    pub timestamp: u64,
    pub value: u64,
    pub write2: bool,
    pub write4: bool,
    pub write8: bool,
}

impl StoreOperation {
    pub fn new(base_address: u64, timestamp: u64, value: u64, bytes: u8) -> Self {
        Self {
            base_address,
            timestamp,
            value,
            write2: bytes == 2,
            write4: bytes == 4,
            write8: bytes == 8,
        }
    }

    /// The 8 `ARE_BYTES[value[i], 0]` range checks this op sends, for the BITWISE
    /// table's multiplicity bookkeeping.
    pub fn collect_bitwise_ops(&self) -> Vec<super::bitwise::BitwiseOperation> {
        use super::bitwise::{BitwiseOperation, BitwiseOperationType};
        (0..8)
            .map(|i| {
                let byte = ((self.value >> (i * 8)) & 0xFF) as u8;
                BitwiseOperation::single_byte(BitwiseOperationType::AreBytes, byte)
            })
            .collect()
    }
}

/// Generates the STORE trace. Each store has a distinct timestamp, so rows are
/// not deduplicated (μ = 1 each); the table is padded to a power of two (min 4).
pub fn generate_store_trace(
    operations: &[StoreOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
        table.set_dword_wl(row_idx, cols::BASE_ADDRESS_0, op.base_address);
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_bool(row_idx, cols::WRITE2, op.write2);
        table.set_bool(row_idx, cols::WRITE4, op.write4);
        table.set_bool(row_idx, cols::WRITE8, op.write8);
        table.set_dword_bl(row_idx, cols::VALUE[0], op.value);
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// All bus interactions for the STORE table:
/// - **Sends** the low-level `MEMW` write (16 elements, no `old`).
/// - **Receives** the high-level `MEMORY` op (flags include the `memory_op` bit).
/// - **Sends** `ARE_BYTES[value[i], 0]` (×8) to range-check the stored bytes.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(10);

    // MEMW[0, base_address, value, timestamp, write2, write4, write8] (write,
    // 16 elements, no `old`).
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(0), // is_register = 0 (memory access)
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::DWordWL,
            },
            // value as 8 individual bytes
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[7],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::WRITE2,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE4,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE8,
                packing: Packing::Direct,
            },
        ],
    ));

    // MEMORY[timestamp, base_address, value, flags] -> 0  (receiver, mult μ).
    // flags = 1 + 4·write2 + 8·write4 + 16·write8 — the `1` is memory_op
    // (matches store.toml).
    interactions.push(BusInteraction::receiver(
        BusId::MemoryOp,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::DWordWL,
            },
            // value cast to DWordWL (8 bytes -> 2 words)
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::DWordBL,
            },
            // flags: memory_op(1) + width bits
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::Column {
                    coefficient: 4,
                    column: cols::WRITE2,
                },
                LinearTerm::Column {
                    coefficient: 8,
                    column: cols::WRITE4,
                },
                LinearTerm::Column {
                    coefficient: 16,
                    column: cols::WRITE8,
                },
            ]),
            // output = 0 (DWordWL): stores write nothing back to rd.
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // ARE_BYTES[value[i], 0] range checks.
    for value_col in cols::VALUE {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: value_col,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
            ],
        ));
    }

    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// Width-flag constraints for the STORE table.
pub struct StoreConstraint {
    constraint_idx: usize,
    kind: StoreConstraintKind,
}

#[derive(Debug, Clone, Copy)]
pub enum StoreConstraintKind {
    /// `write2 + write4 + write8 ∈ {0, 1}` (at most one width bit set).
    WidthSumIsBit,
    /// `(write2 + write4 + write8) ⇒ μ`, i.e. `(Σ width)·(1 − μ) = 0`.
    WidthImpliesMu,
}

impl StoreConstraint {
    pub fn new(kind: StoreConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for StoreConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let w2 = step.get_main_evaluation_element(0, cols::WRITE2).clone();
        let w4 = step.get_main_evaluation_element(0, cols::WRITE4).clone();
        let w8 = step.get_main_evaluation_element(0, cols::WRITE8).clone();
        let sum = &w2 + &w4 + &w8;
        let one = FieldElement::<F>::one();
        match self.kind {
            StoreConstraintKind::WidthSumIsBit => &sum * (&one - &sum),
            StoreConstraintKind::WidthImpliesMu => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                &sum * (&one - &mu)
            }
        }
    }
}

impl Capture for StoreConstraint {
    fn capture(&self, b: &mut IrBuilder) {
        let w2 = b.main(0, cols::WRITE2);
        let w4 = b.main(0, cols::WRITE4);
        let w8 = b.main(0, cols::WRITE8);
        let sum = b.add(w2, w4);
        let sum = b.add(sum, w8);
        let one = b.one();

        let root = match self.kind {
            StoreConstraintKind::WidthSumIsBit => {
                let one_minus_sum = b.sub(one, sum);
                b.mul(sum, one_minus_sum)
            }
            StoreConstraintKind::WidthImpliesMu => {
                let mu = b.main(0, cols::MU);
                let one_minus_mu = b.sub(one, mu);
                b.mul(sum, one_minus_mu)
            }
        };

        b.emit(self.constraint_idx, root);
    }
}

/// Creates all transition constraints for the STORE table: `IS_BIT` on each
/// width flag, the width-sum-is-bit constraint, and width ⇒ μ.
pub fn store_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::new();

    let (is_bit, mut idx) = new_is_bit_constraints(
        &[cols::WRITE2, cols::WRITE4, cols::WRITE8, cols::MU],
        constraint_idx_start,
    );
    for c in is_bit {
        constraints.push(c.boxed());
    }

    constraints.push(StoreConstraint::new(StoreConstraintKind::WidthSumIsBit, idx).boxed());
    idx += 1;
    constraints.push(StoreConstraint::new(StoreConstraintKind::WidthImpliesMu, idx).boxed());
    idx += 1;

    (constraints, idx)
}
