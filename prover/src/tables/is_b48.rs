//! IS_B48 range-check chip.
//!
//! Provides the `IS_B48[W, H]` lookup: `W` is a `Word` and `H` a `Half`, i.e.
//! the 48-bit value `H + 2^16 * W` decomposes into three 16-bit limbs.
//!
//! Spec: `spec/src/is_b48.toml`, `spec/chapters/is_b48.typ`.
//!
//! ## Why the chip exists
//!
//! The `ECALL` accelerators address memory eight bytes at a time and keep one
//! address per access. Those addresses are contiguous — `base`, `base + 8`,
//! `base + 16`, ... — so their top 48 bits are almost always identical, yet each
//! was stored as a `DWordHL` with an `IS_HALF` range check on every one of its
//! four limbs.
//!
//! A caller now range-checks only the bottom limb itself and defers the top 48
//! bits here, which lets it hold the address as a `DWordWHH` (`Half`, `Half`,
//! `Word`) — one column less per address, and two interactions instead of four.
//! KECCAK, the only caller today, goes from 100 `IS_HALF` sends to 25 `IS_HALF`
//! plus 25 `IS_B48`.
//!
//! ## Columns
//! - `value`: three `Half` limbs of the 48-bit value being checked
//! - `μ`: multiplicity — how many callers asked about this value
//!
//! ## How it range-checks
//!
//! BITWISE cannot serve this lookup directly: 2^48 rows do not fit in a table.
//! Instead this chip holds the value split into three `Half` limbs and sends
//! each to BITWISE's `IS_HALF`, whose rows enumerate exactly `[0, 2^16)`. A limb
//! outside that range matches no BITWISE row and the `IS_HALF` bus fails to
//! balance, so every `(W, H)` tuple this chip offers is genuinely a
//! `(Word, Half)` pair. Callers asking about the same 48 bits share one row, so
//! the three `IS_HALF` sends are paid once per *distinct* value rather than once
//! per request.
//!
//! The chip has no polynomial constraints (the spec emits none): a row with
//! `μ = 0` is inert on both sides — it neither consumes `IS_HALF` nor provides
//! `IS_B48` — so nothing needs to pin `μ`, and padding rows cost nothing.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Column indices for IS_B48 table
// =========================================================================

/// Column definitions for the IS_B48 table.
pub mod cols {
    /// value[0]: bits [0, 16) of the checked 48-bit value — the `Half` half of
    /// the bus tuple.
    pub const VALUE_0: usize = 0;
    /// value[1]: bits [16, 32).
    pub const VALUE_1: usize = 1;
    /// value[2]: bits [32, 48).
    pub const VALUE_2: usize = 2;
    /// μ: multiplicity
    pub const MU: usize = 3;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 4;
}

/// Number of bits this chip range-checks in one lookup.
pub const B48_BITS: u32 = 48;

/// One more than the largest value the chip can hold (`2^48`).
pub const B48_LIMIT: u64 = 1 << B48_BITS;

// =========================================================================
// Trace generation
// =========================================================================

/// A single `IS_B48` request: one 48-bit value to range-check.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct B48Operation {
    /// The value being checked. Must be below `2^48`.
    pub value: u64,
}

impl B48Operation {
    /// Create a request for `value`.
    ///
    /// # Panics
    /// If `value >= 2^48`. A caller that lets a larger value through would put a
    /// non-`Half` limb in the trace, which shows up only as an unbalanced
    /// `IS_HALF` bus at proving time; failing here names the real cause.
    pub fn new(value: u64) -> Self {
        assert!(
            value < B48_LIMIT,
            "IS_B48 value {value:#x} does not fit in {B48_BITS} bits"
        );
        Self { value }
    }

    /// The top 48 bits of a 64-bit address, which is what every caller checks.
    pub fn from_address(address: u64) -> Self {
        Self {
            value: address >> 16,
        }
    }

    /// The three `Half` limbs, least significant first.
    pub fn limbs(&self) -> [u16; 3] {
        [
            (self.value & 0xFFFF) as u16,
            ((self.value >> 16) & 0xFFFF) as u16,
            ((self.value >> 32) & 0xFFFF) as u16,
        ]
    }
}

/// Merge requests into one row per distinct value, with `μ` the request count.
///
/// Shared by [`generate_is_b48_trace`] and the trace builder's BITWISE
/// bookkeeping so the two cannot disagree about how many `IS_HALF` lookups this
/// chip sends: both read the same `(value, μ)` pairs.
pub fn merge_operations(operations: &[B48Operation]) -> Vec<(B48Operation, u64)> {
    use std::collections::BTreeMap;

    // BTreeMap, not HashMap: the row order is what the verifier's trace
    // commitment covers, so keeping it a deterministic function of the values
    // makes the trace reproducible across runs.
    let mut op_map: BTreeMap<B48Operation, u64> = BTreeMap::new();
    for op in operations {
        *op_map.entry(*op).or_insert(0) += 1;
    }
    op_map.into_iter().collect()
}

/// Generates the IS_B48 trace from a list of requests.
///
/// Duplicate requests are merged into a single row with summed multiplicity,
/// then padded to the next power of two (minimum 4). Padding rows stay all-zero,
/// so `μ = 0` gates both the `IS_HALF` sends and the `IS_B48` receive.
pub fn generate_is_b48_trace(
    operations: &[B48Operation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let unique_ops = merge_operations(operations);
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, (op, multiplicity)) in unique_ops.iter().enumerate() {
        let [limb_0, limb_1, limb_2] = op.limbs();
        table.set_half(row_idx, cols::VALUE_0, limb_0);
        table.set_half(row_idx, cols::VALUE_1, limb_1);
        table.set_half(row_idx, cols::VALUE_2, limb_2);
        table.set_u64(row_idx, cols::MU, *multiplicity);
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// All bus interactions for the IS_B48 table:
/// - **Sends** `IS_HALF[value[i]]` (×3) to range-check each limb against BITWISE.
/// - **Receives** `IS_B48[value[1] + 2^16*value[2], value[0]]` — the lookup this
///   chip provides, with the top 32 bits as the `Word` and the bottom 16 as the
///   `Half`, matching the spec signature `IS_B48[Word, Half]`.
///
/// All four carry multiplicity `μ`, so a row serving `μ` callers pays for the
/// three limb checks once rather than `μ` times in rows.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(4);

    // IS_HALF[value[i]] for i in 0..3
    for limb_col in [cols::VALUE_0, cols::VALUE_1, cols::VALUE_2] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: limb_col,
                packing: Packing::Direct,
            }],
        ));
    }

    // IS_B48[W, H] -> (nothing): W = value[1] + 2^16*value[2], H = value[0].
    interactions.push(BusInteraction::receiver(
        BusId::IsB48,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::VALUE_1,
                },
                LinearTerm::Column {
                    coefficient: 1 << 16,
                    column: cols::VALUE_2,
                },
            ]),
            BusValue::Packed {
                start_column: cols::VALUE_0,
                packing: Packing::Direct,
            },
        ],
    ));

    interactions
}
