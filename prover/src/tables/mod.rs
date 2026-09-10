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
pub mod blake3;
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
    /// KECCAK_RND, the widest table in the machine: 1,480 main + 740 ext aux.
    ///
    /// Sized by the DEVICE per-chunk budget rather than the host equal-memory
    /// rule above, because this is the table that sets the round-2-to-4 peak:
    /// its incremental is `n · (120·aux + 1232)` = 90,032 B/row at blowup 4, so
    /// one chunk of `H = 5.6 GiB` is 66,800 rows — 30× narrower in rows than
    /// anything the rule above produces, and the reason a uniform ROW cap
    /// mis-allocates by 52× across this registry.
    ///
    /// ⚠ The budget is QUANTISED, and the schedule must land on a power of two.
    /// A chunk pads to `next_pow2(rows)`, so the incremental is set by the
    /// PADDED height and not by the cap: 2,730 permutations and 2,048 both pad
    /// to 2^16 and both cost 5.68 GiB. A cap between two powers of two buys no
    /// device memory and costs sub-proofs. The schedule is therefore
    /// `2^floor(log2(H / (120·aux + 1232)))`, group-aligned UNDER that — here
    /// 1,365 × 24 = 32,760 rows, padding to 2^15 for 2.93 GiB.
    ///
    /// The chunker divides by `ROWS_PER_PERMUTATION` and splits OPERATIONS, so
    /// the alignment is structural rather than a value that has to stay right.
    pub const KECCAK_RND: usize = 1365 * super::keccak_rnd::ROWS_PER_PERMUTATION; // 32,760 -> 2^15
    // Auxiliary ALU / memory / CPU32 dispatch chips
    pub const EQ: usize = 1 << 20;
    pub const BYTEWISE: usize = 1 << 20;
    pub const STORE: usize = 1 << 20;
    pub const CPU32: usize = 1 << 19;
}

/// DEVICE per-chunk ceilings — the cap schedule, in rows, at blowup 4.
///
/// Distinct from [`max_rows`], which sizes chunks for equal HOST memory. This
/// module sizes them for the round-2-to-4 DEVICE incremental,
/// `n · (120·aux + 1232)` bytes, against a single budget `H`. Two numbers, two
/// resources, and they disagree by up to 52× across this registry because
/// `120·aux + 1232` spans 1.75× over the tables `max_rows` covers and 66× over
/// all of them.
///
/// A table appears here only when its device budget is BELOW the tallest
/// posture the uniform knob offers (2^21). At `H = 5.68 GiB` — the budget
/// KECCAK_RND's own cap sets — that is exactly two tables:
///
/// | table | aux | `120·aux+1232` | rows at `H` |
/// |---|---|---|---|
/// | KECCAK_RND | 740 | 90,032 | 2^15, group-aligned to 1,365 × 24 |
/// | DVRM | 17 | 3,272 | 2^20 |
/// | HINT | 14 | 2,912 | 2^21, does NOT bind |
/// | MEMW | 13 | 2,792 | 2^21, does NOT bind |
///
/// ⚠ Only KECCAK_RND is far enough off the uniform knob's scale to need an
/// entry here. Every other table's device budget is within one power of two of
/// the knob, so the knob itself is the right instrument for them: at a 2^22
/// epoch `LAMBDA_VM_MAX_ROWS_LOG2=20` holds the whole population at 2^20 and
/// `H` = 2.93 GiB, where 21 leaves MEMW_A at 5.00 GiB and the epoch does not
/// fit. Enumerating tables here instead would mean enumerating them correctly,
/// and the population is what the knob already covers.
///
/// [`MaxRowsConfig::uniform`] takes the MINIMUM of the knob and these, so the
/// knob can still lower a cap but never raise one past the card. Without that,
/// `LAMBDA_VM_MAX_ROWS_LOG2=21` — the line every run uses — silently hands
/// KECCAK_RND 87,381 permutations per chunk and DVRM 2^21 rows, and the
/// schedule is inert in the only configuration anyone runs.
pub mod device_ceiling {
    /// Permutation-aligned; see [`super::max_rows::KECCAK_RND`].
    pub const KECCAK_RND: usize = super::max_rows::KECCAK_RND;
    /// 17 aux columns is the widest LogUp footprint outside the hash chips,
    /// so DVRM's incremental at 2^21 is 6.54 GiB — it, not KECCAK_RND, would
    /// set `H` if the knob were allowed to raise it there.
    pub const DVRM: usize = 1 << 20;
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
    pub keccak_rnd: usize,
}

impl Default for MaxRowsConfig {
    /// The production values from [`max_rows`], unless
    /// `LAMBDA_VM_MAX_ROWS_LOG2` overrides them with one uniform cap.
    ///
    /// The env knob is a prover-side SHAPE choice, like `TABLE_PARALLELISM` is
    /// a resource one: chunk counts already ride the statement (the verifier
    /// reads them from the proof it checks, never from this config), so two
    /// provers with different caps produce differently-chunked but equally
    /// verifiable epochs. It exists for compression-posture measurement — the
    /// production 2^19/2^20 values are sized for equal-memory parallel chunks,
    /// which multiplies SUB-PROOFS per epoch, and every extra sub-proof is a
    /// leg the recursion wrap pays for. Tall-table postures (2^24) trade chunk
    /// parallelism for fewer legs.
    fn default() -> Self {
        if let Ok(v) = std::env::var("LAMBDA_VM_MAX_ROWS_LOG2") {
            let n: u32 = v
                .parse()
                .expect("LAMBDA_VM_MAX_ROWS_LOG2 must be an integer");
            assert!(
                (5..=26).contains(&n),
                "LAMBDA_VM_MAX_ROWS_LOG2 must be in 5..=26, got {n}"
            );
            return Self::uniform(1 << n);
        }
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
            keccak_rnd: max_rows::KECCAK_RND,
        }
    }
}

impl MaxRowsConfig {
    /// One cap for every table — the tall-table posture the env override uses.
    pub fn uniform(rows: usize) -> Self {
        Self {
            cpu: rows,
            memw: rows,
            memw_aligned: rows,
            dvrm: rows.min(device_ceiling::DVRM),
            mul: rows,
            lt: rows,
            shift: rows,
            load: rows,
            branch: rows,
            memw_register: rows,
            eq: rows,
            bytewise: rows,
            store: rows,
            cpu32: rows,
            // NOT flattened to `rows`: see `device_ceiling`. The uniform knob is
            // a SHAPE choice —
            // it exists to make tables taller and cut the sub-proof count —
            // but KECCAK_RND's cap is a DEVICE budget, and raising it past
            // that budget is the one thing it cannot survive: at a 2^22 epoch
            // an uncapped KECCAK_RND is 2^17 rows and 11.17 GiB of round-2-to-4
            // incremental, against 5.68 GiB capped. `LAMBDA_VM_MAX_ROWS_LOG2=21`
            // would otherwise give it 87,381 permutations per chunk, more than a
            // 2^22 epoch contains, and the cap would be silently inert in every
            // run that sets the knob — which is every campaign run.
            //
            // The knob still LOWERS it: `min` keeps a small-cap posture small.
            keccak_rnd: rows.min(device_ceiling::KECCAK_RND),
        }
    }

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
            keccak_rnd: 1 << 5,
        }
    }
}
