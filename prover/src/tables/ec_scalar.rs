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

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

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
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in ops.iter().enumerate() {
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_wl(row_idx, cols::PTR_0, op.ptr);
        table.set_byte(row_idx, cols::OFFSET, op.offset);
        for i in 0..8 {
            table.set_bool(row_idx, cols::limb_bit(i), ((op.limb >> i) & 1) != 0);
        }
        table.set_bool(row_idx, cols::LAST_LIMB, op.last_limb);
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    // Padding rows keep every field 0: all IS_BIT constraints hold (0 is a bit) and the
    // implication constraints (a·b = 0) hold trivially.
    trace
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

    // 3. Receive Bit[timestamp, 8*offset + i] for each set bit (mult = limb_bits[i]).
    for i in 0..8 {
        let [t0, t1] = ts();
        interactions.push(BusInteraction::receiver(
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
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// One body against the generic `ConstraintBuilder` serves the compiled prover
// folder, the verifier folder and IR capture. Constraint indices 0..20.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// EC_SCALAR transition constraints as a single-source [`ConstraintSet`] (20
/// total). No column configuration needed (the layout is fixed via `cols`).
pub struct EcScalarConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for EcScalarConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0..10: unconditional IS_BIT `x·(1−x)` for
        // [mu, limb_bit(0..8), last_limb], in that column order. Iterator
        // chain, not a Vec: eval runs once per LDE row.
        let bit_cols = core::iter::once(cols::MU)
            .chain((0..8).map(cols::limb_bit))
            .chain(core::iter::once(cols::LAST_LIMB));
        for (i, col) in bit_cols.enumerate() {
            let x = b.main(0, col);
            let one = b.one();
            b.emit_base(i, x.clone() * (one - x));
        }

        // idx 10..18: limb_bit(i) · (1 − mu) = 0.
        for i in 0..8 {
            let a = b.main(0, cols::limb_bit(i));
            let mu = b.main(0, cols::MU);
            let one = b.one();
            b.emit_base(10 + i, a * (one - mu));
        }

        // idx 18: last_limb · (1 − mu) = 0.
        let last_limb = b.main(0, cols::LAST_LIMB);
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(18, last_limb * (one - mu));

        // idx 19: last_limb · offset = 0.
        let last_limb = b.main(0, cols::LAST_LIMB);
        let offset = b.main(0, cols::OFFSET);
        b.emit_base(19, last_limb * offset);
    }
}
