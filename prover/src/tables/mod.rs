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
//! ## Memory Tables
//!
//! - **MEMW**: Memory word read/write table
//! - **LOAD**: Memory load with extension table

pub mod types;

pub mod bitwise;
pub mod cpu;
pub mod load;
pub mod lt;
pub mod memw;
pub mod trace_builder;

pub use types::BusId;
