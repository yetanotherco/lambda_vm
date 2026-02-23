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
//! - **BRANCH**: Branch target calculation table
//! - **HALT**: Single-row halt table
//!
//! ## Memory Tables
//!
//! - **MEMW**: Memory word read/write table
//! - **LOAD**: Memory load with extension table
//! - **PAGE**: Paged memory init/final table (one per used page)
//! - **REGISTER**: Register init/final table (32 registers × 8 bytes = 256 rows)

pub mod types;

pub mod bitwise;
pub mod branch;
pub mod cpu;
pub mod decode;
pub mod dvrm;
pub mod halt;
pub mod load;
pub mod lt;
pub mod memw;
pub mod mul;
pub mod page;
pub mod register;
pub mod trace_builder;

pub use types::BusId;

/// Per-table maximum rows, sized so each chunk uses roughly the same memory.
///
/// Effective width = main_cols + 3 × bus_interactions (extension field = 3× cost).
/// MEMW (effective width 241) at 2^19 is the baseline; other tables are scaled
/// proportionally: max_rows = (241 × 2^19) / effective_width, rounded to 2^N.
///
/// | Table  | Main | Bus | Eff.width | Max rows |
/// |--------|------|-----|-----------|----------|
/// | MEMW   |  70  |  57 |    241    |  2^19    |
/// | CPU    |  74  |  40 |    194    |  2^19    |
/// | DVRM   |  34  |  34 |    136    |  2^19    |
/// | MUL    |  26  |  16 |     74    |  2^20    |
/// | LT     |  15  |   9 |     42    |  2^21    |
/// | LOAD   |  18  |   5 |     33    |  2^21    |
/// | BRANCH |  14  |   6 |     32    |  2^21    |
pub mod max_rows {
    pub const CPU: usize = 1 << 19; // 524,288   — eff. width 194
    pub const MEMW: usize = 1 << 19; // 524,288  — eff. width 241 (baseline)
    pub const DVRM: usize = 1 << 19; // 524,288  — eff. width 136
    pub const MUL: usize = 1 << 20; // 1,048,576 — eff. width 74
    pub const LT: usize = 1 << 21; // 2,097,152  — eff. width 42
    pub const LOAD: usize = 1 << 21; // 2,097,152 — eff. width 33
    pub const BRANCH: usize = 1 << 21; // 2,097,152 — eff. width 32
}
