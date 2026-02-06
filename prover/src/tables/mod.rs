//! 64-bit VM prover tables.
//!
//! This module contains the table definitions for proving 64-bit RISC-V VM execution.
//!
//! ## Tables
//!
//! - **BITWISE**: Precomputed lookup table for bitwise operations (2^20 rows)
//! - **LT**: Less-than comparison table
//! - **CPU**: Main execution table
//! - **DECODE**: Instruction decode table
//!
//! ## Memory Tables
//!
//! - **MEMW**: Memory word read/write table
//! - **LOAD**: Memory load with extension table

pub mod types;

pub mod bitwise;
pub mod branch;
pub mod cpu;
pub mod decode;
pub mod halt;
pub mod load;
pub mod lt;
pub mod memw;
pub mod mul;
pub mod segment;
pub mod trace_builder;

pub use segment::{OpBoundaries, SegmentResult};
pub use types::BusId;
