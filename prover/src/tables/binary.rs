//! Binary AIR — proves `lhs op rhs = res` for op ∈ {AND, OR, XOR}
//! where the CPU dispatches whole-64-bit bitwise ops via `BusId::Binary`.
//!
//! ## Status
//!
//! **Phase 2 step 1: skeleton only.** The eventual column layout (per
//! Phase 2 plan, design (a)) keeps byte cols here — `Binary` does the
//! per-byte AND/OR/XOR sends to the existing BITWISE table, paying that
//! cost only on rows that actually fire instead of every CPU row. For
//! step 1 the AIR is empty: no constraints, no buses, padded zero trace.
//!
//! Subsequent steps:
//!
//! - **Step 5** — column layout (byte cols + op selectors), per-byte
//!   BITWISE senders, `BusId::Binary` receiver, witness gen.
//! - **Step 6** — CPU drops byte cols + per-byte AND/OR/XOR sends; this
//!   AIR's senders take over the byte-level BITWISE traffic.

use stark::lookup::BusInteraction;
use stark::trace::TraceTable;

use super::types::{FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for Binary table (skeleton — final shape lands in step 5)
// =========================================================================

/// Column definitions for the Binary table.
pub mod cols {
    /// Placeholder so the module compiles — replaced in step 5.
    pub const NUM_COLUMNS: usize = 1;
}

// =========================================================================
// Trace generation (skeleton)
// =========================================================================

/// Generates an empty Binary trace.
///
/// **Step 1**: 4-row zero trace, real witness generation in step 5.
pub fn generate_binary_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = 4;
    let data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];
    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions (skeleton)
// =========================================================================

/// Returns the Binary bus interactions.
///
/// **Step 1**: empty. Step 5 adds the `BusId::Binary` receiver and the
/// per-byte BITWISE senders.
pub fn bus_interactions() -> Vec<BusInteraction> {
    Vec::new()
}
