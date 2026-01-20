//! 64-bit VM prover tables.
//!
//! This module contains the table definitions for proving 64-bit RISC-V VM execution.
//!
//! ## Tables
//!
//! - **BITWISE**: Precomputed lookup table for bitwise operations (2^20 rows)
//! - **LT**: Less-than comparison table
//! - **BRANCH**: Branch target computation table
//! - **MUL**: 64-bit multiplication table
//! - **SHIFT**: Shift operations table
//! - **CPU**: Main execution table
//! - **DECODE**: Instruction decode table (dummy - spec not available)
//!
//! ## Deferred (Phase 5)
//!
//! - **MEMW**: Memory word read/write table
//! - **LOAD**: Memory load with extension table

pub mod types;

// Phase 1
pub mod bitwise;

// Phase 2
pub mod lt;
// pub mod branch;
// pub mod mul;
// pub mod shift;

// Phase 3 (to be added)
// pub mod cpu;
// pub mod decode;

// Phase 5 - Deferred (to be added)
// pub mod memw;
// pub mod load;

pub use types::BusId;
