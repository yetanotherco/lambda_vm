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
pub mod keccak;
pub mod keccak_rc;
pub mod keccak_rnd;
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

/// Global chunk-size scale (log2), a left-shift applied uniformly to
/// every per-table `max_rows`. Read from `LAMBDA_MAXROWS_SCALE_LOG2`,
/// falling back to [`DEFAULT_MAXROWS_SCALE_LOG2`] when unset/invalid.
///
/// Bigger chunks → fewer chunks → fewer full StarkProof instances, which
/// cuts the per-instance overhead that scales with chunk count (FRI query
/// phase ~219 queries, 20-bit grinding, OOD/DEEP, proof serialization) and
/// pushes LDE sizes past the phased-FFT threshold into the memory-bound
/// regime where the Bailey four-step FFT wins. Full power-of-two chunks
/// are not padded (only the final partial chunk is), so the cached-LDE
/// peak heap stays ~constant across scales.
///
/// This research branch defaults to a non-zero scale so the automated
/// main-vs-PR bench harness exercises the larger tables without needing to
/// inject an env var. Override to sweep: `LAMBDA_MAXROWS_SCALE_LOG2=0`
/// reproduces production sizing, `=1`/`=2` are intermediate steps.
///
/// Clamped to `0..=4` so a stray value can't request a 16M+× blow-up.
fn maxrows_scale_log2_from_env() -> u32 {
    match std::env::var("LAMBDA_MAXROWS_SCALE_LOG2") {
        Ok(s) => match s.trim().parse::<u32>() {
            Ok(v) => v.min(4),
            Err(_) => DEFAULT_MAXROWS_SCALE_LOG2,
        },
        Err(_) => DEFAULT_MAXROWS_SCALE_LOG2,
    }
}

/// Default chunk-size scale. **Production sizing (0).**
///
/// Bigger tables were benchmarked on fib_8M and regress monotonically:
/// ×4 +16% / ×8 +41% prove time (bigger chunks collapse the table-level
/// parallelism the prover relies on, plus last-chunk padding). The
/// phased FFT does not recover it even in the memory-bound regime, so we
/// keep production chunk sizing. Override with `LAMBDA_MAXROWS_SCALE_LOG2`
/// to reproduce the experiment.
const DEFAULT_MAXROWS_SCALE_LOG2: u32 = 0;

impl Default for MaxRowsConfig {
    fn default() -> Self {
        let shift = maxrows_scale_log2_from_env();
        Self {
            cpu: max_rows::CPU << shift,
            memw: max_rows::MEMW << shift,
            memw_aligned: max_rows::MEMW_A << shift,
            dvrm: max_rows::DVRM << shift,
            mul: max_rows::MUL << shift,
            lt: max_rows::LT << shift,
            shift: max_rows::SHIFT << shift,
            load: max_rows::LOAD << shift,
            branch: max_rows::BRANCH << shift,
            memw_register: max_rows::MEMW_R << shift,
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
