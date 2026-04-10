//! CPU_BITWISE chip — handles AND, OR, XOR instruction execution.
//!
//! Split from the main CPU table to reduce effective width. The CPU table
//! drops 24 bus interactions (AND_BYTE×8, OR_BYTE×8, XOR_BYTE×8), going
//! from 278 to 206 effective width.
//!
//! This chip uses the same 74-column layout as CPU and reuses the same
//! constraints (conditional constraints are automatically satisfied since
//! non-bitwise selectors are always 0).

pub use super::cpu::{cols, generate_cpu_trace, CpuOperation};

use stark::lookup::BusInteraction;

/// Bus interactions for the CPU_BITWISE chip.
///
/// Includes: DECODE, IS_BYTE, AND_BYTE×8, OR_BYTE×8, XOR_BYTE×8, MEMW (register + PC).
/// Excludes: MSB16, ZERO, LT, MUL, DVRM, SHIFT, LOAD, STORE, BRANCH, ECALL.
pub fn bus_interactions() -> Vec<BusInteraction> {
    super::cpu::bus_interactions_bitwise_chip()
}
