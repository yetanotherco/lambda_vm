//! Automatic `StorageMode` selection from an analytical peak-RAM estimate.
//!
//! [`peak_bytes`] models the live working set of `multi_prove_with_mode` as the
//! sum of three things:
//!
//! 1. **Persistent across phase D** — every cached LDE and its main/aux Merkle
//!    tree, summed over all tables. These are built in phase A/C and stay
//!    alive until the chunk that owns them runs in phase D.
//! 2. **Concurrent transient** — composition LDE + composition Merkle + FRI
//!    evals + FRI Merkle + the round-2 `constraint_evaluations` Vec, summed
//!    over the *worst-case chunk*. `multi_prove_with_mode` runs `k =
//!    table_parallelism()` tables of round 2-4 in parallel; only those k have
//!    transient state alive at once.
//! 3. **Domain + LdeTwiddles caches** — one entry per unique
//!    `(trace_length, blowup, coset_offset)`, deduplicated by
//!    `multi_prove_with_mode` (see prover.rs around the
//!    `domain_cache` HashMap).
//!
//! Per-table contributions, given padded row count `N`, blowup `B`, full main
//! columns `C_m` (including precomputed/multiplicity columns kept in the LDE),
//! aux columns `C_a`, and `T` main Merkle trees (1 for the unified path, 2 for
//! the preprocessed path that builds precomputed_tree + mult_tree):
//!
//! ```text
//! main LDE          : N × C_m × 8 × (1+B)              (Goldilocks = 8 B)
//! main Merkle       : T × 2 × N × B × 32               (Keccak256 node = 32 B)
//! aux trace + LDE   : N × C_a × 24 × (1+B)             (cubic ext = 24 B)
//! aux Merkle        : 2 × N × B × 32
//! constraint evals  : N × B × 24                       (round 2 transient)
//! composition LDE   : 2 × N × B × 24                   (d=2 → two parts)
//! composition Merkle: N × B × 32                       (PairKeccak: N/2 leaves)
//! FRI evals         : N × B × 24                       (geometric ≈ 1)
//! FRI Merkle        : N × B × 32                       (geometric ≈ 1)
//! ```
//!
//! `aux_cols = ⌈bus_interactions.len() / 2⌉` — the LogUp committed-pair count
//! used by `Traces::total_auxiliary_field_elements`.
//!
//! Plus state kept alive across the prove call:
//!
//! ```text
//! MemoryState.cells : unique_byte_count × 32   (HashMap<u64, (u8,u64)> ≈ 32 B/entry)
//! Log Vec           : cycle_count × 40         (Log struct = 5 × u64)
//! Instructions map  : decode_rows × 32
//! ```
//!
//! Per-table row counts and the state-driving counters all come from
//! [`TableLengths`] via [`count_table_lengths`], itself derived from the
//! execution logs without allocating any operation vectors.
//!
//! [`count_table_lengths`]: crate::tables::trace_builder::count_table_lengths

use crate::tables::bitwise::{
    NUM_ROWS as BITWISE_ROWS, bus_interactions as bitwise_buses, cols::NUM_COLUMNS as BITWISE_COLS,
};
use crate::tables::branch::{bus_interactions as branch_buses, cols::NUM_COLUMNS as BRANCH_COLS};
use crate::tables::commit::{bus_interactions as commit_buses, cols::NUM_COLUMNS as COMMIT_COLS};
use crate::tables::cpu::{bus_interactions as cpu_buses, cols::NUM_COLUMNS as CPU_COLS};
use crate::tables::decode::{bus_interactions as decode_buses, cols::NUM_COLUMNS as DECODE_COLS};
use crate::tables::dvrm::{bus_interactions as dvrm_buses, cols::NUM_COLUMNS as DVRM_COLS};
use crate::tables::halt::{bus_interactions as halt_buses, cols::NUM_COLUMNS as HALT_COLS};
use crate::tables::load::{bus_interactions as load_buses, cols::NUM_COLUMNS as LOAD_COLS};
use crate::tables::lt::{bus_interactions as lt_buses, cols::NUM_COLUMNS as LT_COLS};
use crate::tables::memw::{bus_interactions as memw_buses, cols::NUM_COLUMNS as MEMW_COLS};
use crate::tables::memw_aligned::{
    bus_interactions as memw_a_buses, cols::NUM_COLUMNS as MEMW_A_COLS,
};
use crate::tables::memw_register::{
    bus_interactions as memw_r_buses, cols::NUM_COLUMNS as MEMW_R_COLS,
};
use crate::tables::mul::{bus_interactions as mul_buses, cols::NUM_COLUMNS as MUL_COLS};
use crate::tables::page::{
    DEFAULT_PAGE_SIZE as PAGE_SIZE, bus_interactions as page_buses, cols::NUM_COLUMNS as PAGE_COLS,
};
use crate::tables::register::{
    NUM_REGISTER_ADDRESSES, bus_interactions as register_buses, cols::NUM_COLUMNS as REGISTER_COLS,
};
use crate::tables::shift::{bus_interactions as shift_buses, cols::NUM_COLUMNS as SHIFT_COLS};
use crate::tables::trace_builder::TableLengths;
use stark::storage_mode::StorageMode;
use sysinfo::System;

const GOLDILOCKS_BYTES: u64 = 8;
const CUBIC_EXT_BYTES: u64 = 24;
const KECCAK_NODE_BYTES: u64 = 32;
const LOG_STRUCT_BYTES: u64 = 40;
const MEMORY_CELL_BYTES: u64 = 32;
const INSTRUCTION_MAP_BYTES_PER_ROW: u64 = 32;

/// Fraction of the effective RAM budget below which `Ram` is kept. The
/// remainder is headroom for the OS, other processes, and allocator
/// fragmentation.
pub const SAFETY_FRACTION_NUM: u64 = 9;
pub const SAFETY_FRACTION_DEN: u64 = 10;

/// `(rows, main_cols, aux_cols, num_main_merkle_trees)` for a single table.
type TableSpec = (u64, u64, u64, u64);

/// Bytes alive for the duration of phase D (LDE columns + main/aux Merkle).
fn persistent_per_table(spec: TableSpec, blowup: u64) -> u64 {
    let (rows, main_cols, aux_cols, main_trees) = spec;
    let main_lde = rows * main_cols * GOLDILOCKS_BYTES * (1 + blowup);
    let aux_lde = rows * aux_cols * CUBIC_EXT_BYTES * (1 + blowup);
    let main_merkle = main_trees * 2 * rows * blowup * KECCAK_NODE_BYTES;
    let aux_merkle = 2 * rows * blowup * KECCAK_NODE_BYTES;
    main_lde + aux_lde + main_merkle + aux_merkle
}

/// Bytes alive only while one chunk of `k` tables is in rounds 2-4. Sums:
/// `constraint_evaluations` (transient mid-round 2), the two LDE-size
/// composition parts (d=2 path, every current AIR), the composition Merkle,
/// and the geometric-sum FRI evals + FRI Merkle.
fn transient_per_table(spec: TableSpec, blowup: u64) -> u64 {
    let (rows, _, _, _) = spec;
    let lde_size = rows * blowup;
    let constraint_evals = lde_size * CUBIC_EXT_BYTES;
    let composition_lde = 2 * lde_size * CUBIC_EXT_BYTES;
    let composition_merkle = lde_size * KECCAK_NODE_BYTES;
    let fri_evals = lde_size * CUBIC_EXT_BYTES;
    let fri_merkle = lde_size * KECCAK_NODE_BYTES;
    constraint_evals + composition_lde + composition_merkle + fri_evals + fri_merkle
}

/// Bytes for one (trace_length, blowup, coset_offset) entry in the prover's
/// Domain/LdeTwiddles cache. Domain holds `trace_roots_of_unity` (N elts) and
/// `lde_roots_of_unity_coset` (N×B). LdeTwiddles holds `inv` (~N), `fwd`
/// (~N×B), and `coset_weights` (N). All in base field (Goldilocks, 8 B).
fn domain_cache_bytes(rows: u64, blowup: u64) -> u64 {
    rows * (3 + 2 * blowup) * GOLDILOCKS_BYTES
}

fn aux_cols(bus_count: usize) -> u64 {
    bus_count.div_ceil(2) as u64
}

/// Build the full per-table table list for a given `TableLengths`. Order
/// matches the order tables are added to `air_trace_pairs` in `prove`.
fn table_specs(lengths: &TableLengths) -> Vec<TableSpec> {
    let bitwise_rows = BITWISE_ROWS as u64;
    let register_rows = NUM_REGISTER_ADDRESSES.next_power_of_two() as u64;
    let halt_rows = 1u64;
    let page_rows = PAGE_SIZE as u64;

    let mut specs = vec![
        (
            lengths.cpu_padded_rows,
            CPU_COLS as u64,
            aux_cols(cpu_buses().len()),
            1,
        ),
        (
            lengths.memw_padded_rows,
            MEMW_COLS as u64,
            aux_cols(memw_buses().len()),
            1,
        ),
        (
            lengths.memw_aligned_padded_rows,
            MEMW_A_COLS as u64,
            aux_cols(memw_a_buses().len()),
            1,
        ),
        (
            lengths.memw_register_padded_rows,
            MEMW_R_COLS as u64,
            aux_cols(memw_r_buses().len()),
            1,
        ),
        (
            lengths.load_padded_rows,
            LOAD_COLS as u64,
            aux_cols(load_buses().len()),
            1,
        ),
        (
            lengths.lt_padded_rows,
            LT_COLS as u64,
            aux_cols(lt_buses().len()),
            1,
        ),
        (
            lengths.shift_padded_rows,
            SHIFT_COLS as u64,
            aux_cols(shift_buses().len()),
            1,
        ),
        (
            lengths.mul_padded_rows,
            MUL_COLS as u64,
            aux_cols(mul_buses().len()),
            1,
        ),
        (
            lengths.dvrm_padded_rows,
            DVRM_COLS as u64,
            aux_cols(dvrm_buses().len()),
            1,
        ),
        (
            lengths.branch_padded_rows,
            BRANCH_COLS as u64,
            aux_cols(branch_buses().len()),
            1,
        ),
        (
            lengths.commit_padded_rows,
            COMMIT_COLS as u64,
            aux_cols(commit_buses().len()),
            1,
        ),
        // BITWISE / DECODE / PAGE / REGISTER take the preprocessed-trace commit
        // path: it extracts ALL columns into the LDE and builds two Merkle trees
        // (precomputed_tree + mult_tree), so main_cols = full NUM_COLUMNS and
        // main_trees = 2.
        (
            bitwise_rows,
            BITWISE_COLS as u64,
            aux_cols(bitwise_buses().len()),
            2,
        ),
        (
            lengths.decode_rows,
            DECODE_COLS as u64,
            aux_cols(decode_buses().len()),
            2,
        ),
        (halt_rows, HALT_COLS as u64, aux_cols(halt_buses().len()), 1),
        (
            register_rows,
            REGISTER_COLS as u64,
            aux_cols(register_buses().len()),
            2,
        ),
    ];
    // Each unique 256 KB page → its own PAGE table at PAGE_SIZE rows.
    for _ in 0..lengths.unique_page_count {
        specs.push((
            page_rows,
            PAGE_COLS as u64,
            aux_cols(page_buses(0).len()),
            2,
        ));
    }
    specs
}

/// Peak RAM estimate in bytes for a proof whose trace shape matches `lengths`.
///
/// `blowup_factor` is `ProofOptions::blowup_factor`. `table_parallelism` is the
/// `k` used by `multi_prove_with_mode` to chunk rounds 2-4; pass
/// `stark::prover::table_parallelism()` so the worst-case-chunk transient term
/// matches the runtime.
pub fn peak_bytes(lengths: &TableLengths, blowup_factor: u8, table_parallelism: usize) -> u64 {
    let blowup = blowup_factor as u64;
    let k = table_parallelism.max(1);
    let specs = table_specs(lengths);

    // Persistent: every table's LDE + main/aux Merkle is alive across phase D.
    let persistent_total: u64 = specs.iter().map(|s| persistent_per_table(*s, blowup)).sum();

    // Transient: only k tables run round 2-4 in parallel. Conservative bound is
    // the top-k tables by transient bytes (worst possible chunk assignment).
    let mut transient_per: Vec<u64> = specs
        .iter()
        .map(|s| transient_per_table(*s, blowup))
        .collect();
    transient_per.sort_unstable_by(|a, b| b.cmp(a));
    let transient_total: u64 = transient_per.iter().take(k).sum();

    // Domain + LdeTwiddles cache: one entry per unique padded-row count
    // (blowup_factor and coset_offset are constant across tables in this
    // codebase, so the unique key collapses to `rows`).
    let mut unique_rows: Vec<u64> = specs.iter().map(|s| s.0).collect();
    unique_rows.sort_unstable();
    unique_rows.dedup();
    let domain_total: u64 = unique_rows
        .iter()
        .map(|&r| domain_cache_bytes(r, blowup))
        .sum();

    // State alive across the prove call (memory cells, log Vec, instruction
    // map). Independent of trace shape.
    let state_total = lengths.unique_byte_count * MEMORY_CELL_BYTES
        + lengths.cycle_count * LOG_STRUCT_BYTES
        + lengths.decode_rows * INSTRUCTION_MAP_BYTES_PER_ROW;

    persistent_total + transient_total + domain_total + state_total
}

/// Effective RAM budget against which the estimate is compared.
///
/// Returns `None` when sysinfo can't read system memory and the user hasn't
/// set a cap. The caller should default to `Disk`: sysinfo fails in
/// stripped-down containers where Ram would OOM. Otherwise the budget is
/// the user's cap (if set), clamped down by what the OS reports available.
pub fn effective_budget(available: Option<u64>, cap: Option<u64>) -> Option<u64> {
    match (cap, available) {
        (Some(c), Some(a)) => Some(c.min(a)),
        (Some(c), None) => Some(c),
        (None, a) => a,
    }
}

/// Pick a storage mode given the estimate and the machine's available RAM.
///
/// Uses 90% of the effective budget as the cutoff so there's headroom for the
/// OS, other processes, and allocator fragmentation. `cap` is an optional
/// user-imposed limit (see `ProofOptions::max_ram_bytes`) which overrides the
/// machine's reported available RAM when smaller.
///
/// When neither `available` nor `cap` is known, defaults to `Disk`: sysinfo
/// fails in stripped-down containers where Ram would OOM. Pass a large
/// `max_ram_bytes` to opt out if you know the machine has enough RAM.
///
/// `available` is a one-shot sample. If a concurrent process allocates between
/// this call and phase 5, this function may pick `Ram` and the prover OOMs.
/// The 90% headroom covers background jitter; under contention, pass
/// `ProofOptions::max_ram_bytes` for a hard cap.
pub fn select_storage_mode(
    estimated: u64,
    available: Option<u64>,
    cap: Option<u64>,
) -> StorageMode {
    let Some(budget) = effective_budget(available, cap) else {
        log::warn!(
            "Auto disk-spill: sysinfo could not read system memory and no cap set, \
             defaulting to Disk. Pass max_ram_bytes if the machine has enough RAM."
        );
        return StorageMode::Disk;
    };
    let threshold = budget.saturating_mul(SAFETY_FRACTION_NUM) / SAFETY_FRACTION_DEN;

    if estimated > threshold {
        StorageMode::Disk
    } else {
        StorageMode::Ram
    }
}

/// Query the OS for available (not total) RAM in bytes. Returns `None` when
/// sysinfo can't read system memory (e.g. inside containers without
/// `/proc/meminfo`); a zero free reading on a near-OOM system returns
/// `Some(0)` so the caller forces Disk instead of falling back to Ram and
/// OOMing.
pub fn available_ram_bytes() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    // total_memory disambiguates: 0 means sysinfo can't read system memory;
    // non-zero means available's value (including 0) is real.
    if sys.total_memory() == 0 {
        None
    } else {
        Some(sys.available_memory())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;
    /// Larger than the table count, so every table lands in the top-k and the
    /// per-table delta in `peak_bytes_per_table_increment_is_exact` is purely
    /// additive.
    const ALL_TABLES: usize = 1_000;

    fn empty_lengths() -> TableLengths {
        TableLengths::default()
    }

    /// Adding rows to a single chunked table must increase `peak_bytes` by
    /// exactly the per-row contribution from the formula in the module doc.
    /// Verifies the per-table breakdown is exact rather than averaged.
    #[test]
    fn peak_bytes_per_table_increment_is_exact() {
        let blowup = 2u8;
        let b = blowup as u64;

        let baseline = peak_bytes(&empty_lengths(), blowup, ALL_TABLES);

        let mut lengths = empty_lengths();
        lengths.cpu_padded_rows = 4;
        let bumped = peak_bytes(&lengths, blowup, ALL_TABLES);

        let cpu_main = CPU_COLS as u64;
        let cpu_aux = cpu_buses().len().div_ceil(2) as u64;
        let per_row_persistent = cpu_main * GOLDILOCKS_BYTES * (1 + b)
            + cpu_aux * CUBIC_EXT_BYTES * (1 + b)
            + 2 * b * KECCAK_NODE_BYTES   // main Merkle (1 tree)
            + 2 * b * KECCAK_NODE_BYTES; // aux Merkle
        let per_row_transient = b * CUBIC_EXT_BYTES   // constraint_evaluations
            + 2 * b * CUBIC_EXT_BYTES                  // composition LDE (2 parts, d=2)
            + b * KECCAK_NODE_BYTES                    // composition Merkle (PairKeccak)
            + b * CUBIC_EXT_BYTES                      // FRI evals (geometric ≈ 1)
            + b * KECCAK_NODE_BYTES; // FRI Merkle (geometric ≈ 1)
        let per_row_domain = (3 + 2 * b) * GOLDILOCKS_BYTES;

        // CPU adds 4 rows of persistent + transient (top-k by ALL_TABLES) +
        // its 4-row Domain entry (a fresh unique key not previously present).
        assert_eq!(
            bumped - baseline,
            4 * (per_row_persistent + per_row_transient + per_row_domain)
        );
    }

    /// Higher blowup_factor should produce a strictly larger estimate.
    #[test]
    fn peak_bytes_scales_with_blowup() {
        let lengths = empty_lengths();
        let two = peak_bytes(&lengths, 2, ALL_TABLES);
        let four = peak_bytes(&lengths, 4, ALL_TABLES);
        let eight = peak_bytes(&lengths, 8, ALL_TABLES);
        assert!(two < four);
        assert!(four < eight);
    }

    /// Lower table_parallelism caps the transient sum to fewer tables, so the
    /// estimate must be monotone in `k`.
    #[test]
    fn peak_bytes_monotone_in_table_parallelism() {
        let lengths = empty_lengths();
        let k1 = peak_bytes(&lengths, 2, 1);
        let k4 = peak_bytes(&lengths, 2, 4);
        let k_all = peak_bytes(&lengths, 2, ALL_TABLES);
        assert!(k1 < k4);
        assert!(k4 <= k_all);
    }

    #[test]
    fn select_ram_when_estimate_below_threshold() {
        // 10 GB estimated, 32 GB available → threshold 28.8 GB → Ram.
        let mode = select_storage_mode(10 * GB, Some(32 * GB), None);
        assert_eq!(mode, StorageMode::Ram);
    }

    #[test]
    fn select_disk_when_estimate_exceeds_threshold() {
        // 30 GB estimated, 32 GB available → threshold 28.8 GB → Disk.
        let mode = select_storage_mode(30 * GB, Some(32 * GB), None);
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn cap_forces_disk_when_smaller_than_available() {
        // 10 GB estimated, 64 GB available (would be Ram), but cap=4 GB
        // → threshold = 4 × 0.9 = 3.6 GB → Disk.
        let mode = select_storage_mode(10 * GB, Some(64 * GB), Some(4 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn cap_ignored_when_larger_than_available() {
        // available=8 GB dominates a cap of 64 GB.
        // threshold = 8 × 0.9 = 7.2 GB, estimate 10 GB → Disk.
        let mode = select_storage_mode(10 * GB, Some(8 * GB), Some(64 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn tiny_cap_always_forces_disk() {
        let mode = select_storage_mode(
            peak_bytes(&empty_lengths(), 2, ALL_TABLES),
            Some(64 * GB),
            Some(1_000_000),
        );
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn unknown_available_with_no_cap_defaults_to_disk() {
        // sysinfo failed and no cap was set. Default to Disk: sysinfo fails
        // in stripped-down containers where Ram would OOM. Pass max_ram_bytes
        // to opt out on a known-sized machine.
        let mode = select_storage_mode(peak_bytes(&empty_lengths(), 2, ALL_TABLES), None, None);
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn unknown_available_with_cap_uses_cap_as_budget() {
        // OS can't report; cap is the whole budget.
        let mode = select_storage_mode(10 * GB, None, Some(4 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }
}
