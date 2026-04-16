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
//! - **MEMW**: Memory word read/write table (unaligned/split-timestamp path, 49 cols, 26 interactions)
//! - **MEMW_A**: Memory word read/write table (aligned fast path, 29 cols, 20 interactions)
//! - **LOAD**: Memory load with extension table
//! - **PAGE**: Paged memory init/final table (one per used page)
//! - **REGISTER**: Register init/final table (32 registers × 8 bytes = 256 rows)

pub mod types;

pub mod bitwise;
pub mod branch;
pub mod commit;
pub mod cpu;
pub mod decode;
pub mod dvrm;
pub mod halt;
pub mod load;
pub mod lt;
pub mod memw;
pub mod memw_aligned;
pub mod memw_register;
pub mod mul;
pub mod page;
pub mod register;
pub mod shift;
pub mod trace_builder;

pub use types::BusId;

/// Per-table maximum rows. All tables use the same limit (2^20) to enable
/// batched commitment: when all chunks share the same maximum height, their
/// LDE domains are identical and columns can be committed in a single tree.
/// All tables use the same max_rows (2^20) to enable batched commitment.
/// When all chunks have the same maximum height, their LDE domains are
/// identical, allowing all columns to be committed in a single Merkle tree.
pub mod max_rows {
    pub const UNIFORM: usize = 1 << 20; // 1,048,576

    pub const CPU: usize = UNIFORM;
    pub const MEMW: usize = UNIFORM;
    pub const MEMW_A: usize = UNIFORM;
    pub const DVRM: usize = UNIFORM;
    pub const MUL: usize = UNIFORM;
    pub const LT: usize = UNIFORM;
    pub const SHIFT: usize = UNIFORM;
    pub const LOAD: usize = UNIFORM;
    pub const BRANCH: usize = UNIFORM;
    pub const MEMW_R: usize = UNIFORM;
}

/// Per-table maximum row limits, configurable for different environments.
///
/// `Default` uses the production values from [`max_rows`].
/// [`MaxRowsConfig::small`] uses 2^5 for low-memory testing.
#[derive(Debug, Clone)]
pub struct MaxRowsConfig {
    pub cpu: usize,
    pub memw: usize,
    pub memw_aligned: usize,
    pub dvrm: usize,
    pub mul: usize,
    pub lt: usize,
    pub shift: usize,
    pub load: usize,
    pub branch: usize,
    pub memw_register: usize,
}

impl Default for MaxRowsConfig {
    fn default() -> Self {
        Self {
            cpu: max_rows::CPU,
            memw: max_rows::MEMW,
            memw_aligned: max_rows::MEMW_A,
            dvrm: max_rows::DVRM,
            mul: max_rows::MUL,
            lt: max_rows::LT,
            shift: max_rows::SHIFT,
            load: max_rows::LOAD,
            branch: max_rows::BRANCH,
            memw_register: max_rows::MEMW_R,
        }
    }
}

impl MaxRowsConfig {
    /// Small limits for low-memory testing. Generates multiple chunks
    /// per table even for tiny programs (~32 rows per chunk).
    pub fn small() -> Self {
        Self {
            cpu: 1 << 5,
            memw: 1 << 5,
            memw_aligned: 1 << 5,
            dvrm: 1 << 5,
            mul: 1 << 5,
            lt: 1 << 5,
            shift: 1 << 5,
            load: 1 << 5,
            branch: 1 << 5,
            memw_register: 1 << 5,
        }
    }
}
