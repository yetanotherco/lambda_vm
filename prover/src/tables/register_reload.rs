//! REGISTER_RELOAD table — bridges large timestamp gaps for register accesses.
//!
//! When a register goes unaccessed for more than [`MAX_REG_GAP`] timestamps,
//! the CPU chip's IS_HALF range check `IS_HALF[TIMESTAMP - PREV_TS ± k]` would
//! overflow. This table inserts intermediate Memory bus prove-old/assume-new
//! pairs that chain the gap in steps of at most [`MAX_REG_GAP`] timestamps,
//! so the final CPU IS_HALF delta always fits in [0, 65535].
//!
//! ## Token model
//!
//! Each row bridges a single step from `old_ts` to `new_ts` for one register,
//! keeping the value unchanged:
//!
//! - **Sender** (prove-old word 0): `(1, 2*reg_idx,   0, old_ts, 0, val_lo)`
//! - **Sender** (prove-old word 1): `(1, 2*reg_idx+1, 0, old_ts, 0, val_hi)`
//! - **Receiver** (assume-new word 0): `(1, 2*reg_idx,   0, new_ts, 0, val_lo)`
//! - **Receiver** (assume-new word 1): `(1, 2*reg_idx+1, 0, new_ts, 0, val_hi)`
//!
//! ## Padding
//!
//! Padding rows use `old_ts = new_ts = 0`, so prove-old and assume-new have
//! identical tokens and cancel in the LogUp sum (net contribution = 0).
//!
//! ## Columns (5 total)
//!
//! | Index | Name    | Description                                |
//! |-------|---------|--------------------------------------------|
//! | 0     | reg_idx | Register index (0–63 for x0–x63, 255=PC) |
//! | 1     | old_ts  | Previous timestamp (prove-old)             |
//! | 2     | new_ts  | Intermediate timestamp (assume-new)        |
//! | 3     | val_lo  | Register value word 0 (low 32 bits)        |
//! | 4     | val_hi  | Register value word 1 (high 32 bits)       |

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    /// reg_idx: register index (Byte; 0–63 for x0–x63, 255 for PC x255)
    pub const REG_IDX: usize = 0;
    /// old_ts: previous timestamp (Word; 32-bit value)
    pub const OLD_TS: usize = 1;
    /// new_ts: intermediate/new timestamp (Word; 32-bit value)
    pub const NEW_TS: usize = 2;
    /// val_lo: register value word 0 (low 32 bits)
    pub const VAL_LO: usize = 3;
    /// val_hi: register value word 1 (high 32 bits)
    pub const VAL_HI: usize = 4;

    /// Total number of columns.
    pub const NUM_COLUMNS: usize = 5;
}

// =========================================================================
// Operation type
// =========================================================================

/// A single register reload step.
///
/// Each step contributes 4 Memory bus tokens: prove-old and assume-new for
/// word 0 and word 1 of the register, all with the same value.
#[derive(Debug, Clone, Copy)]
pub struct RegisterReloadOp {
    /// Register index (0–63 for x0–x63, 255 for PC x255).
    pub reg_idx: u8,
    /// Previous timestamp (prove-old).
    pub old_ts: u64,
    /// Intermediate/new timestamp (assume-new). Must be > old_ts.
    pub new_ts: u64,
    /// Register value word 0 (low 32 bits). Unchanged across the step.
    pub val_lo: u32,
    /// Register value word 1 (high 32 bits). Unchanged across the step.
    pub val_hi: u32,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the REGISTER_RELOAD trace table from reload operations.
///
/// Active rows encode (reg_idx, old_ts, new_ts, val_lo, val_hi).
/// Padding rows are all-zero (old_ts = new_ts = 0 → tokens cancel in LogUp).
pub fn generate_register_reload_trace(
    ops: &[RegisterReloadOp],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_active = ops.len();
    let num_rows = num_active.next_power_of_two().max(4);

    let mut data = vec![FE::from(0u64); num_rows * cols::NUM_COLUMNS];

    for (i, op) in ops.iter().enumerate() {
        let base = i * cols::NUM_COLUMNS;
        data[base + cols::REG_IDX] = FE::from(op.reg_idx as u64);
        data[base + cols::OLD_TS] = FE::from(op.old_ts);
        data[base + cols::NEW_TS] = FE::from(op.new_ts);
        data[base + cols::VAL_LO] = FE::from(op.val_lo as u64);
        data[base + cols::VAL_HI] = FE::from(op.val_hi as u64);
    }
    // Padding rows remain zero-initialized: old_ts = new_ts = 0 → self-canceling tokens.

    TraceTable::new_main(data, cols::NUM_COLUMNS, num_rows)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Returns the 4 Memory bus interactions for the REGISTER_RELOAD table.
///
/// Each row emits:
/// - 2 senders (prove-old for word 0 and word 1 at old_ts)
/// - 2 receivers (assume-new for word 0 and word 1 at new_ts)
///
/// With `Multiplicity::One` for all rows. Padding rows have old_ts = new_ts,
/// so their sender and receiver tokens are identical and cancel to zero.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(4);

    // Compute word addresses from reg_idx:
    //   word 0 address = 2 * reg_idx
    //   word 1 address = 2 * reg_idx + 1
    let addr_w0 = BusValue::linear(vec![LinearTerm::Column {
        coefficient: 2,
        column: cols::REG_IDX,
    }]);
    let addr_w1 = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 2,
            column: cols::REG_IDX,
        },
        LinearTerm::Constant(1),
    ]);

    let old_ts = BusValue::Packed {
        start_column: cols::OLD_TS,
        packing: Packing::Direct,
    };
    let new_ts = BusValue::Packed {
        start_column: cols::NEW_TS,
        packing: Packing::Direct,
    };
    let val_lo = BusValue::Packed {
        start_column: cols::VAL_LO,
        packing: Packing::Direct,
    };
    let val_hi = BusValue::Packed {
        start_column: cols::VAL_HI,
        packing: Packing::Direct,
    };

    // Sender (prove-old) word 0: (1, 2*reg_idx, 0, old_ts, 0, val_lo)
    interactions.push(BusInteraction::sender(
        BusId::Memory,
        Multiplicity::One,
        vec![
            BusValue::constant(1),   // is_register = 1
            addr_w0.clone(),         // addr_lo = 2 * reg_idx
            BusValue::constant(0),   // addr_hi = 0
            old_ts.clone(),          // ts_lo = old_ts
            BusValue::constant(0),   // ts_hi = 0
            val_lo.clone(),          // value = val_lo
        ],
    ));

    // Sender (prove-old) word 1: (1, 2*reg_idx+1, 0, old_ts, 0, val_hi)
    interactions.push(BusInteraction::sender(
        BusId::Memory,
        Multiplicity::One,
        vec![
            BusValue::constant(1),   // is_register = 1
            addr_w1.clone(),         // addr_lo = 2 * reg_idx + 1
            BusValue::constant(0),   // addr_hi = 0
            old_ts,                  // ts_lo = old_ts
            BusValue::constant(0),   // ts_hi = 0
            val_hi.clone(),          // value = val_hi
        ],
    ));

    // Receiver (assume-new) word 0: (1, 2*reg_idx, 0, new_ts, 0, val_lo)
    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::One,
        vec![
            BusValue::constant(1),   // is_register = 1
            addr_w0,                 // addr_lo = 2 * reg_idx
            BusValue::constant(0),   // addr_hi = 0
            new_ts.clone(),          // ts_lo = new_ts
            BusValue::constant(0),   // ts_hi = 0
            val_lo,                  // value = val_lo
        ],
    ));

    // Receiver (assume-new) word 1: (1, 2*reg_idx+1, 0, new_ts, 0, val_hi)
    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::One,
        vec![
            BusValue::constant(1),   // is_register = 1
            addr_w1,                 // addr_lo = 2 * reg_idx + 1
            BusValue::constant(0),   // addr_hi = 0
            new_ts,                  // ts_lo = new_ts
            BusValue::constant(0),   // ts_hi = 0
            val_hi,                  // value = val_hi
        ],
    ));

    interactions
}
