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
//! - **REGISTER**: Register init/final table for x0-x31, x254, and x255 word addresses

pub mod types;

pub mod bitwise;
pub mod branch;
pub mod bytewise;
pub mod commit;
pub mod cpu;
pub mod cpu32;
pub mod decode;
pub mod dvrm;
pub mod ecdas;
pub mod ecsm;
pub mod eq;
pub mod global_memory;
pub mod halt;
pub mod hint;
pub mod keccak;
pub mod keccak_rc;
pub mod keccak_rnd;
pub mod load;
pub mod local_to_global;
pub mod lt;
pub mod memw;
pub mod memw_aligned;
pub mod memw_register;
pub mod mul;
pub mod page;
pub mod register;
pub mod shift;
pub mod store;
pub mod trace_builder;

pub use types::BusId;

/// Blowup factors for which we ship static preprocessed-table commitments
/// (bitwise and keccak_rc), pinned by the `static_commitments_tests` drift
/// suite and emitted by the `compute_static_commitments` binary. Shared
/// between the generator and the drift tests so adding a blowup here cannot
/// silently skip a test.
pub const STATIC_BLOWUP_FACTORS: &[u8] = &[2, 4, 8];

/// Per-table maximum rows, sized so each chunk uses roughly the same memory.
///
/// Effective width = main_cols + 3 × bus_interactions (extension field = 3× cost).
/// MEMW (effective width 127) at 2^19 is the baseline; other tables are scaled
/// proportionally: max_rows = (127 × 2^19) / effective_width, rounded to 2^N.
/// (* MEMW_A formula gives 2^20, but set to 2^19 to match MEMW chunk geometry;
///    benchmarks show better parallel throughput with smaller chunks.)
///
/// | Table   | Main | Bus | Eff.width | Max rows |
/// |---------|------|-----|-----------|----------|
/// | MEMW    |  49  |  26 |    127    |  2^19    |
/// | MEMW_A  |  29  |  20 |     89    |  2^19 *  |
/// | CPU     |  74  |  40 |    194    |  2^19    |
/// | DVRM    |  34  |  34 |    136    |  2^19    |
/// | MUL     |  26  |  16 |     74    |  2^20    |
/// | LT      |  15  |   9 |     42    |  2^20    |
/// | SHIFT   |  27  |  15 |     72    |  2^20    |
/// | LOAD    |  18  |   5 |     33    |  2^20    |
/// | BRANCH  |  14  |   6 |     32    |  2^20    |
/// | MEMW_R  |  10  |   7 |     31    |  2^20    |
///
/// The accelerator chips are sized the same way. They are wide, so their limits
/// are small; `accelerator_max_rows_track_effective_width` pins each width to
/// the AIR so a column or bus added to one of these tables cannot silently leave
/// its limit behind.
///
/// | Table      | Main | Bus  | Eff.width | Max rows |
/// |------------|------|------|-----------|----------|
/// | KECCAK     | 511  |  134 |    913    |  2^16    |
/// | KECCAK_RND | 1480 | 1031 |   4573    |  2^14    |
/// | ECSM       | 667  |  579 |   2404    |  2^15    |
/// | ECDAS      | 521  |  388 |   1685    |  2^15    |
/// | HINT       |  41  |   27 |    122    |  2^19    |
/// | COMMIT     |  19  |   18 |     73    |  2^20    |
pub mod max_rows {
    pub const CPU: usize = 1 << 19; // 524,288   — eff. width 194
    pub const MEMW: usize = 1 << 19; // 524,288  — eff. width 127 (baseline)
    pub const MEMW_A: usize = 1 << 19; // 524,288 — eff. width 89
    pub const DVRM: usize = 1 << 19; // 524,288  — eff. width 136
    pub const MUL: usize = 1 << 20; // 1,048,576 — eff. width 74
    pub const LT: usize = 1 << 20; // 1,048,576  — eff. width 42
    pub const SHIFT: usize = 1 << 20; // 1,048,576 — eff. width 72
    pub const LOAD: usize = 1 << 20; // 1,048,576 — eff. width 33
    pub const BRANCH: usize = 1 << 20; // 1,048,576 — eff. width 32
    pub const MEMW_R: usize = 1 << 20; // 1,048,576 — eff. width 31
    // Auxiliary ALU / memory / CPU32 dispatch chips
    pub const EQ: usize = 1 << 20;
    pub const BYTEWISE: usize = 1 << 20;
    pub const STORE: usize = 1 << 20;
    pub const CPU32: usize = 1 << 19;
    // Accelerator chips. Row counts here are trace rows, not calls: KECCAK_RND
    // emits `keccak_rnd::ROUNDS_PER_OP` rows per permutation and chunks on whole
    // permutations, so its limit is the rounded-down multiple of that.
    pub const KECCAK: usize = 1 << 16;
    pub const KECCAK_RND: usize = 1 << 14;
    pub const ECSM: usize = 1 << 15;
    pub const ECDAS: usize = 1 << 15;
    pub const HINT: usize = 1 << 19;
    pub const COMMIT: usize = 1 << 20;
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
    pub eq: usize,
    pub bytewise: usize,
    pub store: usize,
    pub cpu32: usize,
    pub keccak: usize,
    pub keccak_rnd: usize,
    pub ecsm: usize,
    pub ecdas: usize,
    pub hint: usize,
    pub commit: usize,
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
            eq: max_rows::EQ,
            bytewise: max_rows::BYTEWISE,
            store: max_rows::STORE,
            cpu32: max_rows::CPU32,
            keccak: max_rows::KECCAK,
            keccak_rnd: max_rows::KECCAK_RND,
            ecsm: max_rows::ECSM,
            ecdas: max_rows::ECDAS,
            hint: max_rows::HINT,
            commit: max_rows::COMMIT,
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
            eq: 1 << 5,
            bytewise: 1 << 5,
            store: 1 << 5,
            cpu32: 1 << 5,
            // The accelerator chips keep their production limits. Shrinking them
            // to 2^5 would split a committed output or one ECSM ladder into
            // dozens of sub-proofs, which costs every test that uses this config
            // far more than the extra chunk coverage is worth
            // (`accelerator_chunking_tests` shrinks them where it wants chunks).
            keccak: max_rows::KECCAK,
            keccak_rnd: max_rows::KECCAK_RND,
            ecsm: max_rows::ECSM,
            ecdas: max_rows::ECDAS,
            hint: max_rows::HINT,
            commit: max_rows::COMMIT,
        }
    }
}
