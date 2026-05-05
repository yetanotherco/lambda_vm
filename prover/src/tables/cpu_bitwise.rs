//! CPU_BITWISE chip — handles AND, OR, XOR (and their `*W` 32-bit variants).
//!
//! Phase 4 of the CPU-width reduction plan: peel the bitwise rows out of
//! the unified CPU table so the main CPU table stops paying aux EF cells
//! for the unified `BusId::Bitwise` declarations on every non-bitwise row.
//!
//! The chip reuses the CPU column layout and trace generator; the only
//! thing it changes is the `bus_interactions` filter — only buses an
//! AND/OR/XOR row actually fires are declared here.

pub use super::cpu::{CpuOperation, cols, generate_cpu_trace};

use stark::lookup::BusInteraction;

/// Bus interactions for the CPU_BITWISE chip.
pub fn bus_interactions() -> Vec<BusInteraction> {
    super::cpu::bus_interactions_bitwise_chip()
}
