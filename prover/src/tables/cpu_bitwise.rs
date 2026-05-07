//! CPU_BITWISE chip — handles AND, OR, XOR (and their `*W` 32-bit variants).
//!
//! Phase 4 of the CPU-width reduction plan: peel the bitwise rows out of
//! the unified CPU table so the main CPU table stops paying aux EF cells
//! for the unified `BusId::Bitwise` declarations on every non-bitwise row.
//!
//! The chip reuses the CPU column layout and trace generator; the only
//! thing it changes is the `bus_interactions` filter — only buses an
//! AND/OR/XOR row actually fires are declared here.

pub use super::cpu::{CpuOperation, generate_cpu_trace};

use stark::lookup::BusInteraction;

/// Column layout for the CPU_BITWISE chip.
///
/// For now this re-exports the base CPU `cols` module verbatim so the two
/// chips share an identical layout. Substep C2 will start adding chip-local
/// aux cells (halfword decompositions on base CPU only) and C3 will diverge
/// the layouts entirely (drop ARG1[0..7] and RES[0..7] from base CPU).
/// Giving CPU_BITWISE its own `cols` symbol now decouples the constraint and
/// trace files at the symbol level so those later changes don't have to
/// touch every importer.
pub mod cols {
    pub use super::super::cpu::cols::*;
}

/// Bus interactions for the CPU_BITWISE chip.
pub fn bus_interactions() -> Vec<BusInteraction> {
    super::cpu::bus_interactions_bitwise_chip()
}
