//! CPU_MEMORY chip — handles LOAD and STORE instruction execution.
//!
//! Split from the main CPU table to reduce per-row work. The CPU table
//! drops 2 bus interactions (BusId::Load and M7 STORE), and ~15-20% of
//! rows in typical workloads (memory ops are common in real programs).
//!
//! This chip uses the same 74-column layout as CPU and reuses the same
//! constraints (conditional constraints are automatically satisfied since
//! non-memory selectors are always 0).

pub use super::cpu::{CpuOperation, cols, generate_cpu_trace};

use stark::lookup::BusInteraction;

/// Bus interactions for the CPU_MEMORY chip.
///
/// Includes: DECODE, IS_BYTE, MSB16, MEMW (M1/M3/M5/CM54/M7), LOAD.
/// Excludes: AND/OR/XOR byte ops, ZERO, LT, MUL, DVRM, SHIFT, BRANCH, ECALL.
pub fn bus_interactions() -> Vec<BusInteraction> {
    super::cpu::bus_interactions_memory_chip()
}
