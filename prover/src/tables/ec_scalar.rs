//! EC_SCALAR chip — serves the scalar `k` bit-by-bit to the ECDAS chip.
//!
//! One row per scalar byte (32 rows per ECSM ecall, `offset` counting down 31→0). Each row
//! receives a `ServeK[timestamp, ptr, offset]` token, reads byte `k[offset]` from memory,
//! decomposes it into 8 bits, and sends one `Bit[timestamp, 8*offset + i]` token per set bit
//! (the multiplicity is the bit itself). Unless `last_limb` (offset 0) it recurses by sending
//! `ServeK[timestamp, ptr, offset-1]` — a self-referential bus, like COMMIT's `CommitNextByte`.
//!
//! ## Columns (15 total)
//! - `timestamp`: DWordWL (2) — the ECALL timestamp
//! - `ptr`: DWordWL (2) — address of `k` (= `addr_k`)
//! - `offset`: Byte (1) — index of the scalar byte served by this row
//! - `limb_bits`: Bit[8] (8) — bit decomposition of `k[offset]`
//! - `last_limb`: Bit (1) — whether `offset == 0` (terminates the recursion)
//! - `mu`: Bit (1) — multiplicity (1 for real rows, 0 for padding)
//!
//! `limb = Σ 2^i · limb_bits[i]` is virtual (a linear combination, never stored).

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::constraints::templates::new_is_bit_constraints;

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const PTR_0: usize = 2;
    pub const PTR_1: usize = 3;
    pub const OFFSET: usize = 4;
    /// limb_bits[0..8]
    pub const LIMB_BITS: usize = 5;
    pub const LAST_LIMB: usize = 13;
    pub const MU: usize = 14;

    pub const NUM_COLUMNS: usize = 15;

    #[inline]
    pub const fn limb_bit(i: usize) -> usize {
        LIMB_BITS + i
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// One EC_SCALAR row: serving byte `offset` of the scalar at `ptr`.
#[derive(Debug, Clone)]
pub struct EcScalarOperation {
    pub timestamp: u64,
    pub ptr: u64,
    pub offset: u8,
    pub limb: u8,
    pub last_limb: bool,
}

/// Expands a scalar `k` (little-endian bytes) and its ECALL timestamp / address into the
/// 32 EC_SCALAR rows (offsets 31 down to 0).
pub fn rows_for_scalar(timestamp: u64, addr_k: u64, k: &[u8; 32]) -> Vec<EcScalarOperation> {
    (0..32)
        .rev()
        .map(|offset| EcScalarOperation {
            timestamp,
            ptr: addr_k,
            offset: offset as u8,
            limb: k[offset],
            last_limb: offset == 0,
        })
        .collect()
}

// =========================================================================
// Trace generation
// =========================================================================

pub fn generate_ec_scalar_trace(
    ops: &[EcScalarOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);
        data[base + cols::PTR_0] = FE::from(op.ptr & 0xFFFF_FFFF);
        data[base + cols::PTR_1] = FE::from(op.ptr >> 32);
        data[base + cols::OFFSET] = FE::from(op.offset as u64);
        for i in 0..8 {
            data[base + cols::limb_bit(i)] = FE::from(((op.limb >> i) & 1) as u64);
        }
        data[base + cols::LAST_LIMB] = FE::from(op.last_limb as u64);
        data[base + cols::MU] = FE::one();
    }

    // Padding rows keep every field 0: all IS_BIT constraints hold (0 is a bit) and the
    // implication constraints (a·b = 0) hold trivially.
    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// `limb = Σ 2^i · limb_bits[i]` as a single bus element (used as the byte value in MEMW).
fn limb_value() -> BusValue {
    BusValue::linear(
        (0..8)
            .map(|i| LinearTerm::Column {
                coefficient: 1i64 << i,
                column: cols::limb_bit(i),
            })
            .collect(),
    )
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let ts = || {
        [
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
        ]
    };
    let ptr = || {
        [
            BusValue::Packed {
                start_column: cols::PTR_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::PTR_1,
                packing: Packing::Direct,
            },
        ]
    };

    let mut interactions = Vec::with_capacity(11);

    // 1. Receive ServeK[timestamp, ptr, offset] (mult = mu).
    {
        let [t0, t1] = ts();
        let [p0, p1] = ptr();
        interactions.push(BusInteraction::receiver(
            BusId::ServeK,
            Multiplicity::Column(cols::MU),
            vec![
                t0,
                t1,
                p0,
                p1,
                BusValue::Packed {
                    start_column: cols::OFFSET,
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // 2. MEMW: read byte k[offset] at ptr+offset, timestamp+1, width 1 (mult = mu).
    // CO24 layout: [old[8], is_register, base[2], value[8], ts[2], w2, w4, w8].
    {
        let base_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::PTR_0,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::OFFSET,
            },
        ]);
        let base_hi = BusValue::Packed {
            start_column: cols::PTR_1,
            packing: Packing::Direct,
        };
        let ts_lo_plus_1 = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Constant(1),
        ]);
        let ts_hi = BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        };
        let mut values = Vec::with_capacity(24);
        // old[0..8]: read value = limb, rest 0
        values.push(limb_value());
        for _ in 1..8 {
            values.push(BusValue::constant(0));
        }
        values.push(BusValue::constant(0)); // is_register = 0
        values.push(base_lo);
        values.push(base_hi);
        // value[0..8]: same as old (read)
        values.push(limb_value());
        for _ in 1..8 {
            values.push(BusValue::constant(0));
        }
        values.push(ts_lo_plus_1);
        values.push(ts_hi);
        values.push(BusValue::constant(0)); // w2
        values.push(BusValue::constant(0)); // w4
        values.push(BusValue::constant(0)); // w8 (width 1 byte)
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // 3. Send Bit[timestamp, 8*offset + i] for each set bit (mult = limb_bits[i]).
    for i in 0..8 {
        let [t0, t1] = ts();
        interactions.push(BusInteraction::sender(
            BusId::Bit,
            Multiplicity::Column(cols::limb_bit(i)),
            vec![
                t0,
                t1,
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 8,
                        column: cols::OFFSET,
                    },
                    LinearTerm::Constant(i as i64),
                ]),
            ],
        ));
    }

    // 4. Recurse: send ServeK[timestamp, ptr, offset-1] (mult = mu - last_limb).
    {
        let [t0, t1] = ts();
        let [p0, p1] = ptr();
        interactions.push(BusInteraction::sender(
            BusId::ServeK,
            Multiplicity::Diff(cols::MU, cols::LAST_LIMB),
            vec![
                t0,
                t1,
                p0,
                p1,
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::OFFSET,
                    },
                    LinearTerm::Constant(-1),
                ]),
            ],
        ));
    }

    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// `a · b = 0` or `a · (1 - b) = 0` (degree 2), used for the spec's implication
/// constraints (`limb_bits_i = 1 ⇒ μ = 1`, `last_limb ⇒ μ`, `last_limb ⇒ offset = 0`).
struct MulZeroConstraint {
    a: usize,
    b: usize,
    /// when true, the second factor is `(1 - b)` instead of `b`
    b_complement: bool,
    constraint_idx: usize,
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MulZeroConstraint {
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
        let a = step.get_main_evaluation_element(0, self.a).clone();
        let b = step.get_main_evaluation_element(0, self.b).clone();
        if self.b_complement {
            a * (FieldElement::<F>::one() - b)
        } else {
            a * b
        }
    }
}

/// Creates all EC_SCALAR transition constraints (20 total).
pub fn create_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::with_capacity(20);
    let mut idx = constraint_idx_start;

    // IS_BIT for mu, limb_bits[0..8], last_limb.
    let mut bit_cols = vec![cols::MU];
    bit_cols.extend((0..8).map(cols::limb_bit));
    bit_cols.push(cols::LAST_LIMB);
    let (bit_constraints, next) = new_is_bit_constraints(&bit_cols, idx);
    for c in bit_constraints {
        constraints.push(c.boxed());
    }
    idx = next;

    // limb_bits[i] = 1 ⇒ mu = 1  :  limb_bits[i] · (1 - mu) = 0
    for i in 0..8 {
        constraints.push(
            MulZeroConstraint {
                a: cols::limb_bit(i),
                b: cols::MU,
                b_complement: true,
                constraint_idx: idx,
            }
            .boxed(),
        );
        idx += 1;
    }

    // last_limb = 1 ⇒ mu = 1  :  last_limb · (1 - mu) = 0
    constraints.push(
        MulZeroConstraint {
            a: cols::LAST_LIMB,
            b: cols::MU,
            b_complement: true,
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;

    // last_limb = 1 ⇒ offset = 0  :  last_limb · offset = 0
    constraints.push(
        MulZeroConstraint {
            a: cols::LAST_LIMB,
            b: cols::OFFSET,
            b_complement: false,
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;

    (constraints, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a one-row `TableView` for `row` of the trace (constraints only read row 0).
    fn row_view(
        trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
        row: usize,
    ) -> TableView<GoldilocksField, GoldilocksExtension> {
        let main: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| trace.main_table.get(row, c).clone())
            .collect();
        TableView::new(vec![main], vec![])
    }

    /// Reconstructs each constraint struct (mirrors `create_constraints`) so the test can
    /// evaluate them directly; returns `(value_col_or_pair description, value)` is overkill,
    /// so we just assert every constraint evaluates to zero on every row.
    #[test]
    fn constraints_hold_on_generated_trace() {
        use crate::constraints::templates::IsBitConstraint;

        let mut k = [0u8; 32];
        // a scalar with assorted bit patterns across several bytes
        k[0] = 0b1010_0101;
        k[1] = 0xFF;
        k[15] = 0x80;
        k[31] = 0x01;
        let ops = rows_for_scalar(444, 0x3000, &k);
        let trace = generate_ec_scalar_trace(&ops);

        // IS_BIT columns
        let mut bit_cols = vec![cols::MU];
        bit_cols.extend((0..8).map(cols::limb_bit));
        bit_cols.push(cols::LAST_LIMB);

        for row in 0..trace.num_rows() {
            let view = row_view(&trace, row);
            for &col in &bit_cols {
                let v = IsBitConstraint::unconditional(col, 0).evaluate(&view);
                assert_eq!(v, FE::zero(), "IS_BIT col {col} row {row}");
            }
            // implication constraints
            for i in 0..8 {
                let c = MulZeroConstraint {
                    a: cols::limb_bit(i),
                    b: cols::MU,
                    b_complement: true,
                    constraint_idx: 0,
                };
                assert_eq!(c.evaluate(&view), FE::zero(), "limb_bit{i}=>mu row {row}");
            }
            let c = MulZeroConstraint {
                a: cols::LAST_LIMB,
                b: cols::MU,
                b_complement: true,
                constraint_idx: 0,
            };
            assert_eq!(c.evaluate(&view), FE::zero(), "last_limb=>mu row {row}");
            let c = MulZeroConstraint {
                a: cols::LAST_LIMB,
                b: cols::OFFSET,
                b_complement: false,
                constraint_idx: 0,
            };
            assert_eq!(c.evaluate(&view), FE::zero(), "last_limb=>offset row {row}");
        }
    }

    #[test]
    fn last_limb_set_only_at_offset_zero() {
        let k = [7u8; 32];
        let ops = rows_for_scalar(4, 0x100, &k);
        assert_eq!(ops.len(), 32);
        for op in &ops {
            assert_eq!(op.last_limb, op.offset == 0);
        }
        // 32 distinct offsets 31..0
        assert_eq!(ops[0].offset, 31);
        assert_eq!(ops[31].offset, 0);
    }

    #[test]
    fn create_constraints_count() {
        let (constraints, next) = create_constraints(0);
        assert_eq!(constraints.len(), 20);
        assert_eq!(next, 20);
    }
}
