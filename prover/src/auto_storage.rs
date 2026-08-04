//! Automatic `StorageMode` selection from an analytical peak-RAM estimate.
//!
//! `FORCE_DISK_SPILL` env var forces `StorageMode::Disk` regardless of the
//! estimate.

use crate::tables::bitwise::{
    NUM_ROWS as BITWISE_ROWS, bus_interactions as bitwise_buses, cols::NUM_COLUMNS as BITWISE_COLS,
};
use crate::tables::branch::{bus_interactions as branch_buses, cols::NUM_COLUMNS as BRANCH_COLS};
use crate::tables::commit::{bus_interactions as commit_buses, cols::NUM_COLUMNS as COMMIT_COLS};
use crate::tables::cpu::{bus_interactions as cpu_buses, cols::NUM_COLUMNS as CPU_COLS};
use crate::tables::decode::{bus_interactions as decode_buses, cols::NUM_COLUMNS as DECODE_COLS};
use crate::tables::dma::{bus_interactions as dma_buses, cols::NUM_COLUMNS as DMA_COLS};
use crate::tables::dma_set::{
    bus_interactions as dma_set_buses, cols::NUM_COLUMNS as DMA_SET_COLS,
};
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
use stark::prover::table_parallelism;
use stark::storage_mode::StorageMode;
use sysinfo::System;

pub(crate) const GOLDILOCKS_BYTES: u64 = 8;
pub(crate) const CUBIC_EXT_BYTES: u64 = 24;
pub(crate) const KECCAK_NODE_BYTES: u64 = 32;
const LOG_STRUCT_BYTES: u64 = 40;
const MEMORY_CELL_BYTES: u64 = 32;
const INSTRUCTION_MAP_BYTES_PER_ROW: u64 = 32;

/// 9/10 budget headroom for OS, other processes, and allocator slack.
pub const SAFETY_FRACTION_NUM: u64 = 9;
pub const SAFETY_FRACTION_DEN: u64 = 10;

/// `(rows, main_cols, aux_cols, num_main_merkle_trees)` for a single table.
type TableSpec = (u64, u64, u64, u64);

/// Bytes counted as alive for the whole proof (LDE columns + main/aux Merkle).
/// Deliberately an over-estimate for the aux half — see `peak_bytes`.
fn persistent_per_table(spec: TableSpec, blowup: u64) -> u64 {
    let (rows, main_cols, aux_cols, main_trees) = spec;
    let main_lde = rows
        .saturating_mul(main_cols)
        .saturating_mul(GOLDILOCKS_BYTES)
        .saturating_mul(1 + blowup);
    let aux_lde = rows
        .saturating_mul(aux_cols)
        .saturating_mul(CUBIC_EXT_BYTES)
        .saturating_mul(1 + blowup);
    let main_merkle = main_trees
        .saturating_mul(2)
        .saturating_mul(rows)
        .saturating_mul(blowup)
        .saturating_mul(KECCAK_NODE_BYTES);
    let aux_merkle = if aux_cols > 0 {
        2u64.saturating_mul(rows)
            .saturating_mul(blowup)
            .saturating_mul(KECCAK_NODE_BYTES)
    } else {
        0
    };
    main_lde
        .saturating_add(aux_lde)
        .saturating_add(main_merkle)
        .saturating_add(aux_merkle)
}

/// Bytes (constraint evals, composition, FRI) alive during rounds 2-4 for one chunk.
fn transient_per_table(spec: TableSpec, blowup: u64) -> u64 {
    let (rows, _, _, _) = spec;
    let lde_size = rows.saturating_mul(blowup);
    let constraint_evals = lde_size.saturating_mul(CUBIC_EXT_BYTES);
    let composition_lde = lde_size.saturating_mul(2).saturating_mul(CUBIC_EXT_BYTES);
    let composition_merkle = lde_size.saturating_mul(KECCAK_NODE_BYTES);
    let fri_evals = lde_size.saturating_mul(CUBIC_EXT_BYTES);
    let fri_merkle = lde_size.saturating_mul(KECCAK_NODE_BYTES);
    constraint_evals
        .saturating_add(composition_lde)
        .saturating_add(composition_merkle)
        .saturating_add(fri_evals)
        .saturating_add(fri_merkle)
}

/// Bytes for one Domain/LdeTwiddles cache entry.
fn domain_cache_bytes(rows: u64, blowup: u64) -> u64 {
    rows.saturating_mul(3 + 2 * blowup)
        .saturating_mul(GOLDILOCKS_BYTES)
}

fn aux_cols(bus_count: usize) -> u64 {
    bus_count.div_ceil(2) as u64
}

/// Per-table specs in the same order as `air_trace_pairs` in `prove`.
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
        (
            lengths.dma_padded_rows,
            DMA_COLS as u64,
            aux_cols(dma_buses().len()),
            1,
        ),
        (
            lengths.dma_set_padded_rows,
            DMA_SET_COLS as u64,
            aux_cols(dma_set_buses().len()),
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

/// Estimates heap from `lengths` and `blowup_factor`. Picks `Disk` if the
/// estimate is greater than available RAM, else `Ram`. `FORCE_DISK_SPILL` env
/// var forces `Disk`.
pub fn decide(lengths: &TableLengths, blowup_factor: u8) -> StorageMode {
    if std::env::var("FORCE_DISK_SPILL").is_ok() {
        log::info!("storage_mode: Disk (forced via FORCE_DISK_SPILL)");
        return StorageMode::Disk;
    }
    let estimated = peak_bytes(lengths, blowup_factor, table_parallelism());
    let mode = select_storage_mode(estimated, available_ram_bytes());
    log::info!("estimated_peak_bytes: {estimated}, storage_mode: {mode:?}");
    mode
}

/// Peak RAM estimate in bytes for a proof whose trace shape matches `lengths`.
///
/// `table_parallelism` is the prover's `k` (`stark::prover::table_parallelism`),
/// and it is not only a prover knob: `decide` feeds it in here, so the `cuda`
/// arm's `cores * 2 / 3` doubles the transient term below versus the CPU arm's
/// `cores / 3` and makes `Disk` more likely. That direction is safe (it
/// over-estimates), but it means a change to `k` changes the storage decision.
pub fn peak_bytes(lengths: &TableLengths, blowup_factor: u8, table_parallelism: usize) -> u64 {
    let blowup = blowup_factor as u64;
    let k = table_parallelism.max(1);
    let specs = table_specs(lengths);

    // Persistent: every table's main LDE + Merkle really is alive at once (the
    // Round 1 main commit is a phase-wide barrier). The aux LDE no longer is —
    // it is produced and consumed inside one table's fused task, so at most k
    // coexist — but it is still counted for every table here, which keeps this
    // an over-estimate rather than making the bound unsound.
    let persistent_total: u64 = specs
        .iter()
        .map(|s| persistent_per_table(*s, blowup))
        .fold(0u64, u64::saturating_add);

    // Transient: only k tables run the fused aux+rounds task at a time. The
    // top-k tables by transient bytes bound it; with the scheduler's
    // heaviest-first admission that top-k is also the set actually admitted
    // first, so this is the realistic peak, not a worst case.
    let mut transient_per: Vec<u64> = specs
        .iter()
        .map(|s| transient_per_table(*s, blowup))
        .collect();
    transient_per.sort_unstable_by(|a, b| b.cmp(a));
    let transient_total: u64 = transient_per
        .iter()
        .take(k)
        .copied()
        .fold(0u64, u64::saturating_add);

    // Domain + LdeTwiddles cache: one entry per unique padded-row count
    // (blowup_factor and coset_offset are constant across tables in this
    // codebase, so the unique key collapses to `rows`).
    let mut unique_rows: Vec<u64> = specs.iter().map(|s| s.0).collect();
    unique_rows.sort_unstable();
    unique_rows.dedup();
    let domain_total: u64 = unique_rows
        .iter()
        .map(|&r| domain_cache_bytes(r, blowup))
        .fold(0u64, u64::saturating_add);

    // State alive across the prove call (memory cells, log Vec, instruction
    // map). Independent of trace shape.
    let state_total = lengths
        .unique_byte_count
        .saturating_mul(MEMORY_CELL_BYTES)
        .saturating_add(lengths.cycle_count.saturating_mul(LOG_STRUCT_BYTES))
        .saturating_add(
            lengths
                .decode_rows
                .saturating_mul(INSTRUCTION_MAP_BYTES_PER_ROW),
        );

    persistent_total
        .saturating_add(transient_total)
        .saturating_add(domain_total)
        .saturating_add(state_total)
}

/// `Disk` if `estimated` exceeds `available` minus a safety margin, else
/// `Ram`. Defaults to `Disk` when `available` is `None`.
pub(crate) fn select_storage_mode(estimated: u64, available: Option<u64>) -> StorageMode {
    let Some(available) = available else {
        log::warn!("Auto disk-spill: sysinfo could not read system memory, defaulting to Disk.");
        return StorageMode::Disk;
    };
    let threshold = available.saturating_mul(SAFETY_FRACTION_NUM) / SAFETY_FRACTION_DEN;
    if estimated > threshold {
        StorageMode::Disk
    } else {
        StorageMode::Ram
    }
}

/// OS-available RAM, or `None` if sysinfo can't read it.
fn available_ram_bytes() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    // total_memory == 0 means sysinfo can't read; otherwise available is real.
    if sys.total_memory() == 0 {
        None
    } else {
        Some(sys.available_memory())
    }
}
