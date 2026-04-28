//! Automatic `StorageMode` selection from an analytical peak-RAM estimate.
//!
//! [`peak_bytes`] sums every allocation `multi_prove_with_mode` keeps alive
//! during phase 5 + commitment + composition + FRI, derived from the STARK
//! structure (no regression fit, no averaged coefficients). Per-table:
//!
//! ```text
//! main trace        : rows × main_cols × 8                 (Goldilocks = 8 B)
//! main LDE          : rows × main_cols × 8 × blowup
//! main Merkle       : (2 × rows × blowup) × 32             (Keccak256 node = 32 B)
//! aux trace         : rows × aux_cols × 24                 (cubic ext = 24 B)
//! aux LDE           : rows × aux_cols × 24 × blowup
//! aux Merkle        : (2 × rows × blowup) × 32
//! composition LDE   : rows × blowup × 24                   (one ext-field column)
//! composition Merkle: (2 × rows × blowup) × 32
//! FRI evals         : ~rows × blowup × 24 × 2              (geometric over layers)
//! FRI Merkle        : ~(2 × rows × blowup) × 32 × 2        (geometric over layers)
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
    NUM_PRECOMPUTED_COLS as BITWISE_PRE, NUM_ROWS as BITWISE_ROWS,
    bus_interactions as bitwise_buses, cols::NUM_COLUMNS as BITWISE_COLS,
};
use crate::tables::branch::{bus_interactions as branch_buses, cols::NUM_COLUMNS as BRANCH_COLS};
use crate::tables::commit::{bus_interactions as commit_buses, cols::NUM_COLUMNS as COMMIT_COLS};
use crate::tables::cpu::{bus_interactions as cpu_buses, cols::NUM_COLUMNS as CPU_COLS};
use crate::tables::decode::{
    NUM_PRECOMPUTED_COLS as DECODE_PRE, bus_interactions as decode_buses,
    cols::NUM_COLUMNS as DECODE_COLS,
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
    DEFAULT_PAGE_SIZE as PAGE_SIZE, NUM_PREPROCESSED_COLS as PAGE_PRE,
    bus_interactions as page_buses, cols::NUM_COLUMNS as PAGE_COLS,
};
use crate::tables::register::{
    NUM_PREPROCESSED_COLS as REGISTER_PRE, NUM_REGISTER_ADDRESSES,
    bus_interactions as register_buses, cols::NUM_COLUMNS as REGISTER_COLS,
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

/// Peak RAM estimate in bytes for a proof whose trace shape matches `lengths`.
/// `blowup_factor` is the LDE blowup from `ProofOptions::blowup_factor`.
pub fn peak_bytes(lengths: &TableLengths, blowup_factor: u8) -> u64 {
    let b = blowup_factor as u64;

    // Bytes a single table contributes given its padded-row count, its
    // non-preprocessed main columns, and its aux columns. Sums every
    // allocation listed in the module doc that scales with that table.
    let table_bytes = |rows: u64, main_cols: u64, aux_cols: u64| -> u64 {
        // Trace + LDE: full LDE includes the trace itself, so factor = (1 + B).
        let main_lde = rows * main_cols * GOLDILOCKS_BYTES * (1 + b);
        let aux_lde = rows * aux_cols * CUBIC_EXT_BYTES * (1 + b);
        // Merkle: 2N nodes (binary tree over N = rows × blowup leaves) × 32 B.
        let main_merkle = 2 * rows * b * KECCAK_NODE_BYTES;
        let aux_merkle = 2 * rows * b * KECCAK_NODE_BYTES;
        // Composition: one ext-field column on the LDE domain + its Merkle.
        let composition_lde = rows * b * CUBIC_EXT_BYTES;
        let composition_merkle = 2 * rows * b * KECCAK_NODE_BYTES;
        // FRI evals across all layers (in-place fold halves each layer, but
        // each FriLayer struct keeps its own clone of pre-fold evals; geometric
        // series 1 + 1/2 + 1/4 + … = 2).
        let fri_evals = 2 * rows * b * CUBIC_EXT_BYTES;
        // FRI Merkle uses PairKeccak256 (one leaf per pair of evals), so first
        // layer has rows × blowup / 2 leaves → ~rows × blowup nodes × 32 B.
        // Geometric sum × 2 across layers.
        let fri_merkle = 2 * rows * b * KECCAK_NODE_BYTES;

        main_lde
            + aux_lde
            + main_merkle
            + aux_merkle
            + composition_lde
            + composition_merkle
            + fri_evals
            + fri_merkle
    };

    let aux = |bus_count: usize| -> u64 { bus_count.div_ceil(2) as u64 };

    let bitwise_rows = BITWISE_ROWS as u64;
    let register_rows = NUM_REGISTER_ADDRESSES.next_power_of_two() as u64;
    let halt_rows = 1u64;
    let page_rows = PAGE_SIZE as u64;

    let mut total = 0u64;
    total += table_bytes(
        lengths.cpu_padded_rows,
        CPU_COLS as u64,
        aux(cpu_buses().len()),
    );
    total += table_bytes(
        lengths.memw_padded_rows,
        MEMW_COLS as u64,
        aux(memw_buses().len()),
    );
    total += table_bytes(
        lengths.memw_aligned_padded_rows,
        MEMW_A_COLS as u64,
        aux(memw_a_buses().len()),
    );
    total += table_bytes(
        lengths.memw_register_padded_rows,
        MEMW_R_COLS as u64,
        aux(memw_r_buses().len()),
    );
    total += table_bytes(
        lengths.load_padded_rows,
        LOAD_COLS as u64,
        aux(load_buses().len()),
    );
    total += table_bytes(
        lengths.lt_padded_rows,
        LT_COLS as u64,
        aux(lt_buses().len()),
    );
    total += table_bytes(
        lengths.shift_padded_rows,
        SHIFT_COLS as u64,
        aux(shift_buses().len()),
    );
    total += table_bytes(
        lengths.mul_padded_rows,
        MUL_COLS as u64,
        aux(mul_buses().len()),
    );
    total += table_bytes(
        lengths.dvrm_padded_rows,
        DVRM_COLS as u64,
        aux(dvrm_buses().len()),
    );
    total += table_bytes(
        lengths.branch_padded_rows,
        BRANCH_COLS as u64,
        aux(branch_buses().len()),
    );
    total += table_bytes(
        lengths.commit_padded_rows,
        COMMIT_COLS as u64,
        aux(commit_buses().len()),
    );
    total += table_bytes(
        bitwise_rows,
        (BITWISE_COLS - BITWISE_PRE) as u64,
        aux(bitwise_buses().len()),
    );
    total += table_bytes(
        lengths.decode_rows,
        (DECODE_COLS - DECODE_PRE) as u64,
        aux(decode_buses().len()),
    );
    total += table_bytes(halt_rows, HALT_COLS as u64, aux(halt_buses().len()));
    total += table_bytes(
        register_rows,
        (REGISTER_COLS - REGISTER_PRE) as u64,
        aux(register_buses().len()),
    );
    // Each unique 256 KB page gets its own PAGE table of `PAGE_SIZE` rows.
    let page_main = (PAGE_COLS - PAGE_PRE) as u64;
    let page_aux = aux(page_buses(0).len());
    total += lengths.unique_page_count * table_bytes(page_rows, page_main, page_aux);

    // State kept alive across the prove call.
    total += lengths.unique_byte_count * MEMORY_CELL_BYTES;
    total += lengths.cycle_count * LOG_STRUCT_BYTES;
    total += lengths.decode_rows * INSTRUCTION_MAP_BYTES_PER_ROW;

    total
}

/// Effective RAM budget against which the estimate is compared.
///
/// Returns `None` when the OS can't report available memory and the user
/// hasn't set a cap; in that case the caller should default to `Ram` rather
/// than force Disk on every proof. Otherwise the budget is the user's cap (if
/// set), clamped down by what the OS reports available.
pub fn effective_budget(available: Option<u64>, cap: Option<u64>) -> Option<u64> {
    match (cap, available) {
        (Some(c), Some(a)) => Some(c.min(a)),
        (Some(c), None) => Some(c),
        (None, a) => a,
    }
}

/// Pick a storage mode given the estimate and the machine's available RAM.
///
/// Uses 80% of the effective budget as the cutoff so there's headroom for the
/// OS, other processes, and allocator fragmentation. `cap` is an optional
/// user-imposed limit (see `ProofOptions::max_ram_bytes`) which overrides the
/// machine's reported available RAM when smaller.
///
/// `available` is a one-shot sample. If a concurrent process allocates
/// between this call and phase 5, this function may pick `Ram` and the
/// prover OOMs. The 80% headroom and the estimator's 1.3× margin cover
/// background jitter; under contention, pass `ProofOptions::max_ram_bytes`
/// for a hard cap.
pub fn select_storage_mode(
    estimated: u64,
    available: Option<u64>,
    cap: Option<u64>,
) -> StorageMode {
    const SAFETY_FRACTION_NUM: u64 = 4;
    const SAFETY_FRACTION_DEN: u64 = 5;

    let Some(budget) = effective_budget(available, cap) else {
        return StorageMode::Ram;
    };
    let threshold = budget.saturating_mul(SAFETY_FRACTION_NUM) / SAFETY_FRACTION_DEN;

    if estimated > threshold {
        StorageMode::Disk
    } else {
        StorageMode::Ram
    }
}

/// Query the OS for currently available RAM (not total) in bytes. Returns
/// `None` when the OS can't report a figure (e.g. inside containers without
/// `/proc/meminfo`).
pub fn available_ram_bytes() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    match sys.available_memory() {
        0 => None,
        n => Some(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn empty_lengths() -> TableLengths {
        TableLengths::default()
    }

    /// Adding rows to a single chunked table must increase `peak_bytes` by
    /// exactly `delta_rows × table_bytes(1, main_cols, aux_cols)` from the
    /// formula in the module doc. Verifies the per-table breakdown is exact
    /// rather than an averaged approximation.
    #[test]
    fn peak_bytes_per_table_increment_is_exact() {
        let blowup = 2u8;
        let b = blowup as u64;

        let baseline = peak_bytes(&empty_lengths(), blowup);

        let mut lengths = empty_lengths();
        lengths.cpu_padded_rows = 4;
        let bumped = peak_bytes(&lengths, blowup);
        let cpu_main = CPU_COLS as u64;
        let cpu_aux = cpu_buses().len().div_ceil(2) as u64;
        let per_row = cpu_main * GOLDILOCKS_BYTES * (1 + b)
            + cpu_aux * CUBIC_EXT_BYTES * (1 + b)
            + 2 * 2 * b * KECCAK_NODE_BYTES // main + aux Merkle (Batched: 2N nodes per N leaves = N rows × B)
            + b * CUBIC_EXT_BYTES            // composition LDE
            + 2 * b * KECCAK_NODE_BYTES      // composition Merkle
            + 2 * b * CUBIC_EXT_BYTES        // FRI evals (geometric × 2)
            + 2 * b * KECCAK_NODE_BYTES; // FRI Merkle (Pair backend, geometric × 2)
        assert_eq!(bumped - baseline, 4 * per_row);
    }

    /// Higher blowup_factor should produce a strictly larger estimate.
    #[test]
    fn peak_bytes_scales_with_blowup() {
        let lengths = empty_lengths();
        let two = peak_bytes(&lengths, 2);
        let four = peak_bytes(&lengths, 4);
        let eight = peak_bytes(&lengths, 8);
        assert!(two < four);
        assert!(four < eight);
    }

    #[test]
    fn select_ram_when_estimate_below_threshold() {
        // 10 GB estimated, 32 GB available → threshold 25.6 GB → Ram.
        let mode = select_storage_mode(10 * GB, Some(32 * GB), None);
        assert_eq!(mode, StorageMode::Ram);
    }

    #[test]
    fn select_disk_when_estimate_exceeds_threshold() {
        // 30 GB estimated, 32 GB available → threshold 25.6 GB → Disk.
        let mode = select_storage_mode(30 * GB, Some(32 * GB), None);
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn cap_forces_disk_when_smaller_than_available() {
        // 10 GB estimated, 64 GB available (would be Ram), but cap=4 GB
        // → threshold = 4 × 0.8 = 3.2 GB → Disk.
        let mode = select_storage_mode(10 * GB, Some(64 * GB), Some(4 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn cap_ignored_when_larger_than_available() {
        // available=8 GB dominates a cap of 64 GB.
        // threshold = 8 × 0.8 = 6.4 GB, estimate 10 GB → Disk.
        let mode = select_storage_mode(10 * GB, Some(8 * GB), Some(64 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn tiny_cap_always_forces_disk() {
        let mode = select_storage_mode(
            peak_bytes(&empty_lengths(), 2),
            Some(64 * GB),
            Some(1_000_000),
        );
        assert_eq!(mode, StorageMode::Disk);
    }

    #[test]
    fn unknown_available_with_no_cap_falls_back_to_ram() {
        // OS can't report available memory. Without a cap we can't make an
        // informed decision, so stay in Ram rather than forcing Disk on every
        // proof.
        let mode = select_storage_mode(peak_bytes(&empty_lengths(), 2), None, None);
        assert_eq!(mode, StorageMode::Ram);
    }

    #[test]
    fn unknown_available_with_cap_uses_cap_as_budget() {
        // OS can't report; cap is the whole budget.
        let mode = select_storage_mode(10 * GB, None, Some(4 * GB));
        assert_eq!(mode, StorageMode::Disk);
    }
}
