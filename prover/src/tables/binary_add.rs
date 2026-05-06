//! BinaryAdd AIR — proves `lhs + rhs = sum (mod 2^64)` for ADD-style ops
//! that the CPU dispatches via `BusId::BinaryAdd`.
//!
//! ## Status
//!
//! **Phase 2 step 1: skeleton only.** Column layout and module wiring are
//! in place; no transition constraints, no bus interactions, no real
//! witness generation. The AIR proves trivially against a padded zero
//! trace and absorbs nothing. Subsequent steps fill in:
//!
//! - **Step 2** — carry-chain transition constraints + receiver on
//!   `BusId::BinaryAdd` for ADD/LOAD ops.
//! - **Step 3** — extend the receiver to STORE/SUB/BEQ/JALR.
//! - **Step 4** — drop the now-redundant inline carry constraints from
//!   the CPU AIR.
//!
//! ## Eventual column layout (target after step 2)
//!
//! | Range | Cols | Description |
//! |---|---:|---|
//! | `LHS_LO, LHS_HI` | 2 | lhs as DWordWL |
//! | `RHS_LO, RHS_HI` | 2 | rhs as DWordWL |
//! | `SUM_LO, SUM_HI` | 2 | sum as DWordWL |
//! | `CARRY_0, CARRY_1` | 2 | bit (carry between word boundaries) |
//! | `MU_ADD, MU_SUB` | 2 | per-flavour multiplicities |
//! | **NUM_COLUMNS** | **10** | |

use stark::lookup::BusInteraction;
use stark::trace::TraceTable;

use super::types::{FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for BinaryAdd table
// =========================================================================

/// Column definitions for the BinaryAdd table.
pub mod cols {
    /// lhs (low word) — DWordWL[0]
    pub const LHS_LO: usize = 0;
    /// lhs (high word) — DWordWL[1]
    pub const LHS_HI: usize = 1;

    /// rhs (low word)
    pub const RHS_LO: usize = 2;
    /// rhs (high word)
    pub const RHS_HI: usize = 3;

    /// sum = lhs + rhs (low word)
    pub const SUM_LO: usize = 4;
    /// sum (high word)
    pub const SUM_HI: usize = 5;

    /// Bit: carry from `LHS_LO + RHS_LO` into the high word.
    pub const CARRY_0: usize = 6;
    /// Bit: overflow carry from `LHS_HI + RHS_HI + CARRY_0` (discarded by mod 2^64).
    pub const CARRY_1: usize = 7;

    /// Multiplicity for the forward-add flavour (ADD/LOAD/STORE/JALR send here).
    pub const MU_ADD: usize = 8;
    /// Multiplicity for the reverse-add flavour (SUB/BEQ send here, with operands swapped).
    pub const MU_SUB: usize = 9;

    /// Total column count.
    pub const NUM_COLUMNS: usize = 10;
}

// =========================================================================
// Trace generation (skeleton)
// =========================================================================

/// Generates an empty BinaryAdd trace.
///
/// **Step 1**: returns a 4-row zero trace (the minimum the framework
/// accepts). Real witness generation lands in step 2 alongside the
/// receiver and constraints.
pub fn generate_binary_add_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = 4;
    let data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];
    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions (skeleton)
// =========================================================================

/// Returns the BinaryAdd bus interactions.
///
/// **Step 1**: empty — AIR absorbs nothing yet. Step 2 adds the
/// `BusId::BinaryAdd` receiver.
pub fn bus_interactions() -> Vec<BusInteraction> {
    Vec::new()
}
