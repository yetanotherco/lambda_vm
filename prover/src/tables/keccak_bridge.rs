//! KECCAK_BRIDGE — carries one keccak permutation request across the
//! epoch/global proof boundary.
//!
//! KECCAK_RND is a pure function: it touches no memory, has no timestamp
//! ordering, and carries no state between calls. So it does not have to be
//! proved once per continuation epoch — one run-wide instance can serve every
//! epoch's requests. This table is what makes that split possible.
//!
//! One row per keccak permutation call, holding the request's `(timestamp,
//! input_state, output_state)`. The same trace is committed in TWO proofs, with
//! opposite bus polarity in each:
//!
//! - **Epoch proof** ([`epoch_bus_interactions`]): stands in for the 24-round
//!   chain. It RECEIVES `(timestamp, 0, input_state)` from the KECCAK core and
//!   SENDS `(timestamp, 24, output_state)` back to it, closing the epoch-local
//!   `Keccak` bus. KECCAK_RND and KECCAK_RC leave the epoch's table set.
//! - **Global proof** ([`global_bus_interactions`]): SENDS `(timestamp, 0,
//!   input_state)` and RECEIVES `(timestamp, 24, output_state)`, handing the
//!   request to the single run-wide KECCAK_RND chain.
//!
//! ## Why this needs no constraints
//!
//! The bridge is a pure conduit — no range checks, no transition constraints.
//! Its columns must simultaneously match the KECCAK core's state bytes on the
//! epoch-local bus and the round chain's state bytes on the global bus, and the
//! *same committed columns* feed both (the two proofs' commitments are tied by
//! comparing their Merkle roots). Both partners already range-check every state
//! byte, so the bridge inherits that. This is exactly the argument
//! `local_to_global` relies on for its cross-epoch columns.
//!
//! ## Why no epoch label
//!
//! LogUp is a multiset argument, so if two epochs each request the permutation
//! of the same state, the global chain must serve it twice — multiplicity is
//! preserved. Keccak-f is a pure function, so provenance and ordering are
//! irrelevant, and a timestamp that collides across epochs is harmless: both
//! requests carry the same input and therefore the same output.
//!
//! ## Padding
//!
//! Real rows carry `MU = 1`, power-of-two padding rows `MU = 0`. Every
//! interaction is gated by `Multiplicity::Column(MU)`, so padding rows fire
//! nothing. Dropping a real row breaks the KECCAK core's bus balance in the
//! epoch proof, so `MU` is self-enforced.

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::keccak::KeccakOperation;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Column indices (403 columns)
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    /// in_state[5][5][8] = 200 bytes — the permutation input.
    pub const IN: usize = 2;

    /// out_state[5][5][8] = 200 bytes — the permutation output.
    pub const OUT: usize = IN + 200; // 202

    /// Multiplicity flag (1 for real rows, 0 for padding).
    pub const MU: usize = OUT + 200; // 402

    pub const NUM_COLUMNS: usize = MU + 1; // 403

    /// Index into in_state[x][y][byte]. Same layout as `keccak::cols::input_state`.
    #[inline]
    pub const fn in_state(x: usize, y: usize, byte: usize) -> usize {
        IN + (x + 5 * y) * 8 + byte
    }

    /// Index into out_state[x][y][byte]. Same layout as `keccak::cols::output_state`.
    #[inline]
    pub const fn out_state(x: usize, y: usize, byte: usize) -> usize {
        OUT + (x + 5 * y) * 8 + byte
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// One row per keccak call, in the order the calls were made. For a
/// continuation the caller passes one epoch's ops to build that epoch's
/// bridge; the global proof rebuilds the identical per-epoch traces.
pub fn generate_keccak_bridge_trace(
    ops: &[KeccakOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in ops.iter().enumerate() {
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        for x in 0..5 {
            for y in 0..5 {
                table.set_dword_bl(row_idx, cols::in_state(x, y, 0), op.input[x + 5 * y]);
                table.set_dword_bl(row_idx, cols::out_state(x, y, 0), op.output[x + 5 * y]);
            }
        }
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    // Padding rows are all-zero with MU = 0, so they fire no interaction.
    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

fn packed(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// `(timestamp, round, state[200])` — the `Keccak` bus tuple, matching
/// `keccak::bus_interactions` and `keccak_rnd::bus_interactions` exactly:
/// 200 Byte elements, each its own bus element (no packing).
fn tuple(round: u64, state_base: usize) -> Vec<BusValue> {
    let mut values = vec![
        packed(cols::TIMESTAMP_0),
        packed(cols::TIMESTAMP_1),
        BusValue::constant(round),
    ];
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                values.push(packed(state_base + (x + 5 * y) * 8 + b));
            }
        }
    }
    values
}

/// Epoch-local polarity: absorbs the KECCAK core's request and answers it, so
/// the epoch's `Keccak` bus balances without KECCAK_RND present.
pub fn epoch_bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    vec![
        BusInteraction::receiver(BusId::Keccak, mu(), tuple(0, cols::IN)),
        BusInteraction::sender(BusId::Keccak, mu(), tuple(24, cols::OUT)),
    ]
}

/// Global polarity: replays the request onto the run-wide KECCAK_RND chain.
/// Exactly [`epoch_bus_interactions`] with sender/receiver swapped.
pub fn global_bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    vec![
        BusInteraction::sender(BusId::Keccak, mu(), tuple(0, cols::IN)),
        BusInteraction::receiver(BusId::Keccak, mu(), tuple(24, cols::OUT)),
    ]
}
